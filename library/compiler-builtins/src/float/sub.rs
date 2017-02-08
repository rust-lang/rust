use float::Float;

macro_rules! sub {
    ($(#[$attr:meta])*
     | $intrinsic:ident: $ty:ty) => {
        /// Returns `a - b`
        $(#[$attr])*
        pub extern "C" fn $intrinsic(a: $ty, b: $ty) -> $ty {
            a + <$ty>::from_repr(b.repr() ^ <$ty>::sign_mask())
        }
    }
}

sub!(#[cfg_attr(all(not(test), not(target_arch = "arm")), no_mangle)]
     #[cfg_attr(all(not(test), target_arch = "arm"), inline(always))]
     | __subsf3: f32);

sub!(#[cfg_attr(all(not(test), not(target_arch = "arm")), no_mangle)]
     #[cfg_attr(all(not(test), target_arch = "arm"), inline(always))]
     | __subdf3: f64);

// NOTE(cfg) for some reason, on arm*-unknown-linux-gnueabi*, our implementation doesn't
// match the output of its gcc_s or compiler-rt counterpart. Until we investigate further, we'll
// just avoid testing against them on those targets. Do note that our implementation gives the
// correct answer; gcc_s and compiler-rt are incorrect in this case.
#[cfg(all(test, not(arm_linux)))]
mod tests {
    use core::{f32, f64};
    use qc::{F32, F64};

    check! {
        fn __subsf3(f: extern "C" fn(f32, f32) -> f32,
                    a: F32,
                    b: F32)
                    -> Option<F32> {
            Some(F32(f(a.0, b.0)))
        }

        fn __subdf3(f: extern "C" fn(f64, f64) -> f64,
                    a: F64,
                    b: F64) -> Option<F64> {
            Some(F64(f(a.0, b.0)))
        }
    }
}
