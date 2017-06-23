/// Returns `a` raised to the power `b`
macro_rules! pow {
    ($a: expr, $b: expr) => ({
        let (mut a, mut b) = ($a, $b);
        let recip = b < 0;
        let mut r = 1.0;
        loop {
            if (b & 1) != 0 {
                r *= a;
            }
            b = b.checked_div(2).unwrap_or_else(|| ::abort());
            if b == 0 {
                break;
            }
            a *= a;
        }

        if recip {
            1.0 / r
        } else {
            r
        }
    })
}

intrinsics! {
    pub extern "C" fn __powisf2(a: f32, b: i32) -> f32 {
        pow!(a, b)
    }

    pub extern "C" fn __powidf2(a: f64, b: i32) -> f64 {
        pow!(a, b)
    }
}
