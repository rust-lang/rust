macro_rules! pow {
    ($intrinsic:ident: $fty:ty, $ity:ident) => {
        /// Returns `a` raised to the power `b`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $fty, b: $ity) -> $fty {
            let (mut a, mut b) = (a, b);
            let recip = b < 0;
            let mut r: $fty = 1.0;
            loop {
                if (b & 1) != 0 {
                    r *= a;
                }
                b = sdiv!($ity, b, 2);
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
        }
    }
}

pow!(__powisf2: f32, i32);
pow!(__powidf2: f64, i32);

#[cfg(test)]
mod tests {
    use qc::{I32, F32, F64};

    check! {
        fn __powisf2(f: extern fn(f32, i32) -> f32, 
                     a: F32,
                     b: I32) -> Option<F32> {
            Some(F32(f(a.0, b.0)))
        }

        fn __powidf2(f: extern fn(f64, i32) -> f64, 
                     a: F64,
                     b: I32) -> Option<F64> {
            Some(F64(f(a.0, b.0)))
        }
    }
}
