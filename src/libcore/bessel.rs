// PORT import module that is based on cmath::c_double here
// (cant do better via libm; bessel functions only exist for c_double)

// code that wants to use bessel functions should use
// values of type bessel::t and cast from/to float/f32/f64
// when working with them at the peril of precision loss
// for platform neutrality

import f64::*;

