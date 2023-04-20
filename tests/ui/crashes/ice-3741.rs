//@aux-build:proc_macro_crash.rs:proc-macro

#![warn(clippy::suspicious_else_formatting)]

extern crate proc_macro_crash;
use proc_macro_crash::macro_test;

fn main() {
    macro_test!(2);
}
