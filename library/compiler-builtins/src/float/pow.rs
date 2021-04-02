use float::Float;
use int::Int;

/// Returns `a` raised to the power `b`
fn pow<F: Float>(a: F, b: i32) -> F {
    let mut a = a;
    let recip = b < 0;
    let mut pow = i32::abs_diff(b, 0);
    let mut mul = F::ONE;
    loop {
        if (pow & 1) != 0 {
            mul *= a;
        }
        pow >>= 1;
        if pow == 0 {
            break;
        }
        a *= a;
    }

    if recip {
        F::ONE / mul
    } else {
        mul
    }
}

intrinsics! {
    pub extern "C" fn __powisf2(a: f32, b: i32) -> f32 {
        pow(a, b)
    }

    pub extern "C" fn __powidf2(a: f64, b: i32) -> f64 {
        pow(a, b)
    }
}
