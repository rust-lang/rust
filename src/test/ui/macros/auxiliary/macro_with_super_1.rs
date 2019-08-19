#![crate_type = "lib"]

#[macro_export]
macro_rules! declare {
    () => (
        pub fn aaa() {}

        pub mod bbb {
            use super::aaa;

            pub fn ccc() {
                aaa();
            }
        }
    )
}
