#![feature(super_let)]

#[macro_export]
macro_rules! foo {
    () => {
        {
            super let args = 1;
            &args
        }
    };
}
