package main

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:simd"
import "core:strconv"
import "core:time"
import "base:intrinsics"

PI :: 3.141592653589793
SOLAR_MASS :: 4.0 * PI * PI
DAYS_PER_YEAR :: 365.24
TOTAL_BODIES :: 5
SIMD_LANES :: 4

// Stack-allocated SOA layout
Body_SOA_Stack :: struct #align(32) {
    pos_x:   [TOTAL_BODIES]f64,
    pos_y:   [TOTAL_BODIES]f64,
    pos_z:   [TOTAL_BODIES]f64,
    vel_x:   [TOTAL_BODIES]f64,
    vel_y:   [TOTAL_BODIES]f64,
    vel_z:   [TOTAL_BODIES]f64,
    mass:    [TOTAL_BODIES]f64,
}

tmp_x := #simd[SIMD_LANES]f64{}
tmp_y := #simd[SIMD_LANES]f64{}
tmp_z := #simd[SIMD_LANES]f64{}
dt_vec4 := #simd[SIMD_LANES]f64{}

init_system_soa_stack :: #force_inline proc() -> Body_SOA_Stack {
    sys := Body_SOA_Stack{}
    
    // Sun
    sys.pos_x[0] = 0.0
    sys.pos_y[0] = 0.0
    sys.pos_z[0] = 0.0
    sys.vel_x[0] = 0.0
    sys.vel_y[0] = 0.0
    sys.vel_z[0] = 0.0
    sys.mass[0] = SOLAR_MASS
    
    // Jupiter
    sys.pos_x[1] = 4.84143144246472090e+00
    sys.pos_y[1] = -1.16032004402742839e+00
    sys.pos_z[1] = -1.03622044471123109e-01
    sys.vel_x[1] = 1.66007664274403694e-03 * DAYS_PER_YEAR
    sys.vel_y[1] = 7.69901118419740425e-03 * DAYS_PER_YEAR
    sys.vel_z[1] = -6.90460016972063023e-05 * DAYS_PER_YEAR
    sys.mass[1] = 9.54791938424326609e-04 * SOLAR_MASS
    
    // Saturn
    sys.pos_x[2] = 8.34336671824457987e+00
    sys.pos_y[2] = 4.12479856412430479e+00
    sys.pos_z[2] = -4.03523417114321381e-01
    sys.vel_x[2] = -2.76742510726862411e-03 * DAYS_PER_YEAR
    sys.vel_y[2] = 4.99852801234917238e-03 * DAYS_PER_YEAR
    sys.vel_z[2] = 2.30417297573763929e-05 * DAYS_PER_YEAR
    sys.mass[2] = 2.85885980666130812e-04 * SOLAR_MASS
    
    // Uranus
    sys.pos_x[3] = 1.28943695621391310e+01
    sys.pos_y[3] = -1.51111514016986312e+01
    sys.pos_z[3] = -2.23307578892655734e-01
    sys.vel_x[3] = 2.96460137564761618e-03 * DAYS_PER_YEAR
    sys.vel_y[3] = 2.37847173959480950e-03 * DAYS_PER_YEAR
    sys.vel_z[3] = -2.96589568540237556e-05 * DAYS_PER_YEAR
    sys.mass[3] = 4.36624404335156298e-05 * SOLAR_MASS
    
    // Neptune
    sys.pos_x[4] = 1.53796971148509165e+01
    sys.pos_y[4] = -2.59193146099879641e+01
    sys.pos_z[4] = 1.79258772950371181e-01
    sys.vel_x[4] = 2.68067772490389322e-03 * DAYS_PER_YEAR
    sys.vel_y[4] = 1.62824170038242295e-03 * DAYS_PER_YEAR
    sys.vel_z[4] = -9.51592254519715870e-05 * DAYS_PER_YEAR
    sys.mass[4] = 5.15138902046611451e-05 * SOLAR_MASS
    
    return sys
}

advance_stack_simd :: #force_inline proc(sys: ^Body_SOA_Stack, dt: f64, n: int) {
    dt_vec4 = #simd[SIMD_LANES]f64{dt, dt, dt, dt}
    
    for _ in 0..<n {
        #unroll for i in 0..<TOTAL_BODIES {
            j_start := i + 1
            j_count := TOTAL_BODIES - j_start

            if j_count >= SIMD_LANES {
                pos_x_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_x[j_start:j_start+SIMD_LANES])
                pos_y_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_y[j_start:j_start+SIMD_LANES])
                pos_z_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_z[j_start:j_start+SIMD_LANES])
                mass_j  := simd.from_slice(#simd[SIMD_LANES]f64, sys.mass[j_start:j_start+SIMD_LANES])

                pos_x_i := #simd[SIMD_LANES]f64{sys.pos_x[i], sys.pos_x[i], sys.pos_x[i], sys.pos_x[i]}
                pos_y_i := #simd[SIMD_LANES]f64{sys.pos_y[i], sys.pos_y[i], sys.pos_y[i], sys.pos_y[i]}
                pos_z_i := #simd[SIMD_LANES]f64{sys.pos_z[i], sys.pos_z[i], sys.pos_z[i], sys.pos_z[i]}
                mass_i  := #simd[SIMD_LANES]f64{sys.mass[i], sys.mass[i], sys.mass[i], sys.mass[i]}

                dx := simd.sub(pos_x_i, pos_x_j)
                dy := simd.sub(pos_y_i, pos_y_j)
                dz := simd.sub(pos_z_i, pos_z_j)

                dist2 := simd.add(simd.mul(dx, dx), simd.add(simd.mul(dy, dy), simd.mul(dz, dz)))
                distance := simd.sqrt(dist2)
                denom := simd.mul(distance, dist2)
                mag := dt_vec4 / denom

                fx := simd.mul(dx, mag)
                fy := simd.mul(dy, mag)
                fz := simd.mul(dz, mag)

                vel_x_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_x[j_start:j_start+SIMD_LANES])
                vel_y_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_y[j_start:j_start+SIMD_LANES])
                vel_z_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_z[j_start:j_start+SIMD_LANES])
                vel_x_j = simd.fma(fx, mass_i, vel_x_j)
                vel_y_j = simd.fma(fy, mass_i, vel_y_j)
                vel_z_j = simd.fma(fz, mass_i, vel_z_j)
                vel_x_arr := simd.to_array(vel_x_j)
                vel_y_arr := simd.to_array(vel_y_j)
                vel_z_arr := simd.to_array(vel_z_j)
                #unroll for l in 0..<SIMD_LANES {
                    sys.vel_x[j_start + l] = vel_x_arr[l]
                    sys.vel_y[j_start + l] = vel_y_arr[l]
                    sys.vel_z[j_start + l] = vel_z_arr[l]
                }

                // Horizontal reduction
                contrib_x := simd.mul(fx, mass_j)
                contrib_y := simd.mul(fy, mass_j)
                contrib_z := simd.mul(fz, mass_j)

                sys.vel_x[i] -= simd.reduce_add_ordered(contrib_x)
                sys.vel_y[i] -= simd.reduce_add_ordered(contrib_y)
                sys.vel_z[i] -= simd.reduce_add_ordered(contrib_z)
            } else {
                // Scalar fallback
                #no_bounds_check for j in j_start..<TOTAL_BODIES {
                    dx := sys.pos_x[i] - sys.pos_x[j]
                    dy := sys.pos_y[i] - sys.pos_y[j]
                    dz := sys.pos_z[i] - sys.pos_z[j]
                    distance_square := dx*dx + dy*dy + dz*dz
                    distance := math.sqrt_f64(distance_square)
                    mag := dt / (distance * distance_square)
                    fx := dx * mag
                    fy := dy * mag
                    fz := dz * mag
                    sys.vel_x[i] -= fx * sys.mass[j]
                    sys.vel_x[j] += fx * sys.mass[i]
                    sys.vel_y[i] -= fy * sys.mass[j]
                    sys.vel_y[j] += fy * sys.mass[i]
                    sys.vel_z[i] -= fz * sys.mass[j]
                    sys.vel_z[j] += fz * sys.mass[i]
                }
            }
        }

        // SIMD position updates
        pos_x_vec := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_x[0:SIMD_LANES])
        vel_x_vec := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_x[0:SIMD_LANES])
        pos_x_vec = simd.fma(vel_x_vec, dt_vec4, pos_x_vec)
        pos_x_arr := simd.to_array(pos_x_vec)
        #unroll for l in 0..<SIMD_LANES {
            sys.pos_x[l] = pos_x_arr[l]
        }
        sys.pos_x[SIMD_LANES] = simd.fused_mul_add(sys.vel_x[SIMD_LANES], dt, sys.pos_x[SIMD_LANES])

        pos_y_vec := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_y[0:SIMD_LANES])
        vel_y_vec := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_y[0:SIMD_LANES])
        pos_y_vec = simd.fma(vel_y_vec, dt_vec4, pos_y_vec)
        pos_y_arr := simd.to_array(pos_y_vec)

        #unroll for l in 0..<SIMD_LANES {
            sys.pos_y[l] = pos_y_arr[l]
        }
        sys.pos_y[SIMD_LANES] = simd.fused_mul_add(sys.vel_y[SIMD_LANES], dt, sys.pos_y[SIMD_LANES])

        pos_z_vec := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_z[0:SIMD_LANES])
        vel_z_vec := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_z[0:SIMD_LANES])
        pos_z_vec = simd.fma(vel_z_vec, dt_vec4, pos_z_vec)
        pos_z_arr := simd.to_array(pos_z_vec)
        #unroll for l in 0..<SIMD_LANES {
            sys.pos_z[l] = pos_z_arr[l]
        }
        sys.pos_z[SIMD_LANES] = simd.fused_mul_add(sys.vel_z[SIMD_LANES], dt, sys.pos_z[SIMD_LANES])
    }
}

energy_soa_stack :: #force_inline proc(sys: ^Body_SOA_Stack) -> f64 {
    e := 0.0
    #unroll for i in 0..<TOTAL_BODIES {
        speed2 := sys.vel_x[i]*sys.vel_x[i] + sys.vel_y[i]*sys.vel_y[i] + sys.vel_z[i]*sys.vel_z[i]
        e += 0.5 * sys.mass[i] * speed2

        #no_bounds_check for j in (i + 1)..<TOTAL_BODIES {
            dx := sys.pos_x[i] - sys.pos_x[j]
            dy := sys.pos_y[i] - sys.pos_y[j]
            dz := sys.pos_z[i] - sys.pos_z[j]
            distance := math.sqrt_f64(dx*dx + dy*dy + dz*dz)
            e -= sys.mass[i] * sys.mass[j] / distance
        }
    }
    return e
}

energy_soa_stack_simd :: #force_inline proc(sys: ^Body_SOA_Stack) -> f64 {
    e := 0.0
    #unroll for i in 0..<TOTAL_BODIES {
        // kinetic term
        speed2 := sys.vel_x[i]*sys.vel_x[i] + sys.vel_y[i]*sys.vel_y[i] + sys.vel_z[i]*sys.vel_z[i]
        e += 0.5 * sys.mass[i] * speed2

        j_start := i + 1
        j_count := TOTAL_BODIES - j_start
        if j_count >= SIMD_LANES {
            pos_x_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_x[j_start:j_start+SIMD_LANES])
            pos_y_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_y[j_start:j_start+SIMD_LANES])
            pos_z_j := simd.from_slice(#simd[SIMD_LANES]f64, sys.pos_z[j_start:j_start+SIMD_LANES])
            mass_j  := simd.from_slice(#simd[SIMD_LANES]f64, sys.mass[j_start:j_start+SIMD_LANES])

            pos_x_i := #simd[SIMD_LANES]f64{sys.pos_x[i], sys.pos_x[i], sys.pos_x[i], sys.pos_x[i]}
            pos_y_i := #simd[SIMD_LANES]f64{sys.pos_y[i], sys.pos_y[i], sys.pos_y[i], sys.pos_y[i]}
            pos_z_i := #simd[SIMD_LANES]f64{sys.pos_z[i], sys.pos_z[i], sys.pos_z[i], sys.pos_z[i]}

            dx := simd.sub(pos_x_i, pos_x_j)
            dy := simd.sub(pos_y_i, pos_y_j)
            dz := simd.sub(pos_z_i, pos_z_j)

            dist2 := simd.add(simd.mul(dx, dx), simd.add(simd.mul(dy, dy), simd.mul(dz, dz)))
            distance := simd.sqrt(dist2)

            contrib := ( #simd[SIMD_LANES]f64{sys.mass[i], sys.mass[i], sys.mass[i], sys.mass[i]} * mass_j ) / distance
            e -= simd.reduce_add_ordered(contrib)
        } else {
            #no_bounds_check for j in j_start..<TOTAL_BODIES {
                dx := sys.pos_x[i] - sys.pos_x[j]
                dy := sys.pos_y[i] - sys.pos_y[j]
                dz := sys.pos_z[i] - sys.pos_z[j]
                distance := math.sqrt_f64(dx*dx + dy*dy + dz*dz)
                e -= sys.mass[i] * sys.mass[j] / distance
            }
        }
    }
    return e
}

offset_momentum_soa_stack :: #force_inline proc(sys: ^Body_SOA_Stack) {
    px, py, pz := 0.0, 0.0, 0.0
    #unroll for i in 0..<TOTAL_BODIES {
        px -= sys.vel_x[i] * sys.mass[i]
        py -= sys.vel_y[i] * sys.mass[i]
        pz -= sys.vel_z[i] * sys.mass[i]
    }
    sys.vel_x[0] = px / SOLAR_MASS
    sys.vel_y[0] = py / SOLAR_MASS
    sys.vel_z[0] = pz / SOLAR_MASS
}

offset_momentum_soa_stack_simd :: #force_inline proc(sys: ^Body_SOA_Stack) {
    i := 0
    for ; i+SIMD_LANES <= TOTAL_BODIES; i += SIMD_LANES {
        vel_x_v := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_x[i:i+SIMD_LANES])
        vel_y_v := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_y[i:i+SIMD_LANES])
        vel_z_v := simd.from_slice(#simd[SIMD_LANES]f64, sys.vel_z[i:i+SIMD_LANES])
        mass_v  := simd.from_slice(#simd[SIMD_LANES]f64, sys.mass[i:i+SIMD_LANES])

        tmp_x = simd.sub(tmp_x, simd.mul(vel_x_v, mass_v))
        tmp_y = simd.sub(tmp_y, simd.mul(vel_y_v, mass_v))
        tmp_z = simd.sub(tmp_z, simd.mul(vel_z_v, mass_v))
    }

    px := simd.reduce_add_ordered(tmp_x)
    py := simd.reduce_add_ordered(tmp_y)
    pz := simd.reduce_add_ordered(tmp_z)

    #no_bounds_check for k in i..<TOTAL_BODIES {
        px -= sys.vel_x[k] * sys.mass[k]
        py -= sys.vel_y[k] * sys.mass[k]
        pz -= sys.vel_z[k] * sys.mass[k]
    }

    sys.vel_x[0] = px / SOLAR_MASS
    sys.vel_y[0] = py / SOLAR_MASS
    sys.vel_z[0] = pz / SOLAR_MASS
}

main :: proc() {
    args := runtime.args__
    n := strconv.parse_int(auto_cast args[1]) or_else 1000
    
    sys := init_system_soa_stack()
    offset_momentum_soa_stack_simd(&sys)
    
    fmt.printf("simd   energy:  %.9f\n", energy_soa_stack_simd(&sys))

    start := time.tick_now()
    advance_stack_simd(&sys, 0.01, n)
    duration := time.tick_since(start)

    fmt.printf("simd   energy:  %.9f\n", energy_soa_stack_simd(&sys))
    fmt.eprintf("Odin time: %.3f µs\n", f64(duration) / f64(time.Microsecond))
}
