// The `collapse_debuginfo` makes the scopes differ.
#[collapse_debuginfo(yes)]
macro_rules! define_functions {
    () => {
        #[inline(always)]
        fn callee(x: u32) -> u32 {
            x + 1
        }

        #[inline(never)]
        pub fn from_macro(a: u32) -> u32 {
            callee(a)
        }
    };
}
