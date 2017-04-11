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
