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

advance_stack :: #force_inline proc(sys: ^Body_SOA_Stack, dt: f64, n: int) {
    for _ in 0..<n {
        // Force calculations
        #unroll for i in 0..<TOTAL_BODIES {
            #no_bounds_check for j in (i + 1)..<TOTAL_BODIES {
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
        
        // SIMD position updates - vectorize 4 elements at once
        dt_vec := #simd[4]f64{dt, dt, dt, dt}
        
        pos_x_vec := simd.from_slice(#simd[4]f64, sys.pos_x[0:4])
        vel_x_vec := simd.from_slice(#simd[4]f64, sys.vel_x[0:4])
        pos_x_vec += vel_x_vec * dt_vec
        pos_x_arr := simd.to_array(pos_x_vec)
        sys.pos_x[0] = pos_x_arr[0]
        sys.pos_x[1] = pos_x_arr[1]
        sys.pos_x[2] = pos_x_arr[2]
        sys.pos_x[3] = pos_x_arr[3]
        sys.pos_x[4] += sys.vel_x[4] * dt
        
        pos_y_vec := simd.from_slice(#simd[4]f64, sys.pos_y[0:4])
        vel_y_vec := simd.from_slice(#simd[4]f64, sys.vel_y[0:4])
        pos_y_vec += vel_y_vec * dt_vec
        pos_y_arr := simd.to_array(pos_y_vec)
        sys.pos_y[0] = pos_y_arr[0]
        sys.pos_y[1] = pos_y_arr[1]
        sys.pos_y[2] = pos_y_arr[2]
        sys.pos_y[3] = pos_y_arr[3]
        sys.pos_y[4] += sys.vel_y[4] * dt
        
        pos_z_vec := simd.from_slice(#simd[4]f64, sys.pos_z[0:4])
        vel_z_vec := simd.from_slice(#simd[4]f64, sys.vel_z[0:4])
        pos_z_vec += vel_z_vec * dt_vec
        pos_z_arr := simd.to_array(pos_z_vec)
        sys.pos_z[0] = pos_z_arr[0]
        sys.pos_z[1] = pos_z_arr[1]
        sys.pos_z[2] = pos_z_arr[2]
        sys.pos_z[3] = pos_z_arr[3]
        sys.pos_z[4] += sys.vel_z[4] * dt
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

main :: proc() {
    args := runtime.args__
    n := strconv.parse_int(auto_cast args[1]) or_else 1000
    
    sys := init_system_soa_stack()
    offset_momentum_soa_stack(&sys)

    fmt.printf("simd   energy:  %.9f\n", energy_soa_stack(&sys))

    advance_stack(&sys, 0.01, n)

    fmt.printf("simd   energy:  %.9f\n", energy_soa_stack(&sys))
}
