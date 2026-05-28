// rustfmt-newline_style: Windows

#[macro_export]
macro_rules! hmmm___ffi_error {
    ($result:ident) => {
        pub struct $result {
            success: bool,
        }

        impl $result {
            pub fn foo(self) {}
        }
    };
}
