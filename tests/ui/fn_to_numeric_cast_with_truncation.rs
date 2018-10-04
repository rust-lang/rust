// only-64bit
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

    // These should not lint, because because casting to these types
    // does not truncate the function pointer address.
    let _ = foo as u64;
    let _ = foo as i64;
    let _ = foo as u128;
    let _ = foo as i128;
}

fn main() {}
