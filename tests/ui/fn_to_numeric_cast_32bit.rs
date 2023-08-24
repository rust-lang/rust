//@ignore-64bit

#![warn(clippy::fn_to_numeric_cast, clippy::fn_to_numeric_cast_with_truncation)]

fn foo() -> String {
    String::new()
}

fn test_function_to_numeric_cast() {
    let _ = foo as i8;
    //~^ ERROR: casting function pointer `foo` to `i8`, which truncates the value
    //~| NOTE: `-D clippy::fn-to-numeric-cast-with-truncation` implied by `-D warnings`
    let _ = foo as i16;
    //~^ ERROR: casting function pointer `foo` to `i16`, which truncates the value
    let _ = foo as i32;
    //~^ ERROR: casting function pointer `foo` to `i32`, which truncates the value
    let _ = foo as i64;
    //~^ ERROR: casting function pointer `foo` to `i64`
    //~| NOTE: `-D clippy::fn-to-numeric-cast` implied by `-D warnings`
    let _ = foo as i128;
    //~^ ERROR: casting function pointer `foo` to `i128`
    let _ = foo as isize;
    //~^ ERROR: casting function pointer `foo` to `isize`

    let _ = foo as u8;
    //~^ ERROR: casting function pointer `foo` to `u8`, which truncates the value
    let _ = foo as u16;
    //~^ ERROR: casting function pointer `foo` to `u16`, which truncates the value
    let _ = foo as u32;
    //~^ ERROR: casting function pointer `foo` to `u32`, which truncates the value
    let _ = foo as u64;
    //~^ ERROR: casting function pointer `foo` to `u64`
    let _ = foo as u128;
    //~^ ERROR: casting function pointer `foo` to `u128`

    // Casting to usize is OK and should not warn
    let _ = foo as usize;

    // Cast `f` (a `FnDef`) to `fn()` should not warn
    fn f() {}
    let _ = f as fn();
}

fn test_function_var_to_numeric_cast() {
    let abc: fn() -> String = foo;

    let _ = abc as i8;
    //~^ ERROR: casting function pointer `abc` to `i8`, which truncates the value
    let _ = abc as i16;
    //~^ ERROR: casting function pointer `abc` to `i16`, which truncates the value
    let _ = abc as i32;
    //~^ ERROR: casting function pointer `abc` to `i32`, which truncates the value
    let _ = abc as i64;
    //~^ ERROR: casting function pointer `abc` to `i64`
    let _ = abc as i128;
    //~^ ERROR: casting function pointer `abc` to `i128`
    let _ = abc as isize;
    //~^ ERROR: casting function pointer `abc` to `isize`

    let _ = abc as u8;
    //~^ ERROR: casting function pointer `abc` to `u8`, which truncates the value
    let _ = abc as u16;
    //~^ ERROR: casting function pointer `abc` to `u16`, which truncates the value
    let _ = abc as u32;
    //~^ ERROR: casting function pointer `abc` to `u32`, which truncates the value
    let _ = abc as u64;
    //~^ ERROR: casting function pointer `abc` to `u64`
    let _ = abc as u128;
    //~^ ERROR: casting function pointer `abc` to `u128`

    // Casting to usize is OK and should not warn
    let _ = abc as usize;
}

fn fn_with_fn_args(f: fn(i32) -> i32) -> i32 {
    f as i32
    //~^ ERROR: casting function pointer `f` to `i32`, which truncates the value
}

fn main() {}
