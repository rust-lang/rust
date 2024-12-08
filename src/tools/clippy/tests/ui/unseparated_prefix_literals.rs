//@aux-build:proc_macro_derive.rs

#![warn(clippy::unseparated_literal_suffix)]
#![allow(dead_code)]

#[macro_use]
extern crate proc_macro_derive;

// Test for proc-macro attribute
#[derive(ClippyMiniMacroTest)]
struct Foo;

macro_rules! lit_from_macro {
    () => {
        42usize
    };
}

fn main() {
    let _ok1 = 1234_i32;
    let _ok2 = 1234_isize;
    let _ok3 = 0x123_isize;
    let _fail1 = 1234i32;
    let _fail2 = 1234u32;
    let _fail3 = 1234isize;
    let _fail4 = 1234usize;
    let _fail5 = 0x123isize;

    let _okf1 = 1.5_f32;
    let _okf2 = 1_f32;
    let _failf1 = 1.5f32;
    let _failf2 = 1f32;

    // Test for macro
    let _ = lit_from_macro!();

    // Counter example
    let _ = line!();
    // Because `assert!` contains `line!()` macro.
    assert_eq!(4897u32, 32223);
}
