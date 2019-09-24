/// A simple static assertion macro.
#[macro_export]
#[allow_internal_unstable(type_ascription)]
macro_rules! static_assert {
    ($test:expr) => {
        // Use the bool to access an array such that if the bool is false, the access
        // is out-of-bounds.
        #[allow(dead_code)]
        const _: () = [()][!($test: bool) as usize];
    }
}

/// Type size assertion. The first argument is a type and the second argument is its expected size.
#[macro_export]
macro_rules! static_assert_size {
    ($ty:ty, $size:expr) => {
        const _: [(); $size] = [(); ::std::mem::size_of::<$ty>()];
    }
}
