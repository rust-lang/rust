//@stderr-per-bitwidth
//@no-rustfix
#![warn(clippy::fn_to_numeric_cast, clippy::fn_to_numeric_cast_with_truncation)]

fn foo() -> String {
    String::new()
}

fn test_function_to_numeric_cast() {
    let _ = foo as i8;
    let _ = foo as i16;
    let _ = foo as i32;
    let _ = foo as i64;
    let _ = foo as i128;
    let _ = foo as isize;

    let _ = foo as u8;
    let _ = foo as u16;
    let _ = foo as u32;
    let _ = foo as u64;
    let _ = foo as u128;

    // Casting to usize is OK and should not warn
    let _ = foo as usize;

    // Cast `f` (a `FnDef`) to `fn()` should not warn
    fn f() {}
    let _ = f as fn();
}

fn test_function_var_to_numeric_cast() {
    let abc: fn() -> String = foo;

    let _ = abc as i8;
    let _ = abc as i16;
    let _ = abc as i32;
    let _ = abc as i64;
    let _ = abc as i128;
    let _ = abc as isize;

    let _ = abc as u8;
    let _ = abc as u16;
    let _ = abc as u32;
    let _ = abc as u64;
    let _ = abc as u128;

    // Casting to usize is OK and should not warn
    let _ = abc as usize;
}

fn fn_with_fn_args(f: fn(i32) -> i32) -> i32 {
    f as i32
}

fn main() {}
