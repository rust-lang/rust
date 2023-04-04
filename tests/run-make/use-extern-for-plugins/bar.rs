#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]
#![crate_name = "a"]

#[macro_export]
macro_rules! bar {
    () => ()
}
