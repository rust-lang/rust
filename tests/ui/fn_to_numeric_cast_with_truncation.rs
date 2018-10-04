#![feature(tool_lints)]

#![warn(clippy::fn_to_numeric_cast_with_truncation)]
#![allow(clippy::fn_to_numeric_cast)]

fn foo() -> String { String::new() }

fn test_fn_to_numeric_cast_with_truncation() {
    let _ = foo as i8;
    let _ = foo as i16;
    let _ = foo as i32;
    let _ = foo as u8;
    let _ = foo as u16;
    let _ = foo as u32;

    // TODO: Is it bad to have these tests?
    // Running the tests on a different architechture will
    // produce different results
    let _ = foo as u64;
    let _ = foo as i64;
}

fn main() {}
