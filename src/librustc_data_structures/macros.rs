/// A simple static assertion macro. The first argument should be a unique
/// ALL_CAPS identifier that describes the condition.
#[macro_export]
#[allow_internal_unstable(type_ascription)]
macro_rules! static_assert {
    ($name:ident: $test:expr) => {
        // Use the bool to access an array such that if the bool is false, the access
        // is out-of-bounds.
        #[allow(dead_code)]
        static $name: () = [()][!($test: bool) as usize];
    }
}
