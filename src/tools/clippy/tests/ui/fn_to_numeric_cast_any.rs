#![warn(clippy::fn_to_numeric_cast_any)]
#![allow(clippy::fn_to_numeric_cast, clippy::fn_to_numeric_cast_with_truncation)]
//@no-rustfix
fn foo() -> u8 {
    0
}

fn generic_foo<T>(x: T) -> T {
    x
}

trait Trait {
    fn static_method() -> u32 {
        2
    }
}

struct Struct;

impl Trait for Struct {}

fn fn_pointer_to_integer() {
    let _ = foo as i8;
    //~^ fn_to_numeric_cast_any

    let _ = foo as i16;
    //~^ fn_to_numeric_cast_any

    let _ = foo as i32;
    //~^ fn_to_numeric_cast_any

    let _ = foo as i64;
    //~^ fn_to_numeric_cast_any

    let _ = foo as i128;
    //~^ fn_to_numeric_cast_any

    let _ = foo as isize;
    //~^ fn_to_numeric_cast_any

    let _ = foo as u8;
    //~^ fn_to_numeric_cast_any

    let _ = foo as u16;
    //~^ fn_to_numeric_cast_any

    let _ = foo as u32;
    //~^ fn_to_numeric_cast_any

    let _ = foo as u64;
    //~^ fn_to_numeric_cast_any

    let _ = foo as u128;
    //~^ fn_to_numeric_cast_any

    let _ = foo as usize;
    //~^ fn_to_numeric_cast_any
}

fn static_method_to_integer() {
    let _ = Struct::static_method as usize;
    //~^ fn_to_numeric_cast_any
}

fn fn_with_fn_arg(f: fn(i32) -> u32) -> usize {
    f as usize
    //~^ fn_to_numeric_cast_any
}

fn fn_with_generic_static_trait_method<T: Trait>() -> usize {
    T::static_method as usize
    //~^ fn_to_numeric_cast_any
}

fn closure_to_fn_to_integer() {
    let clos = |x| x * 2_u32;

    let _ = (clos as fn(u32) -> u32) as usize;
    //~^ fn_to_numeric_cast_any
}

fn fn_to_raw_ptr() {
    let _ = foo as *const ();
    //~^ fn_to_numeric_cast_any
}

fn cast_fn_to_self() {
    // Casting to the same function pointer type should be permitted.
    let _ = foo as fn() -> u8;
}

fn cast_generic_to_concrete() {
    // Casting to a more concrete function pointer type should be permitted.
    let _ = generic_foo as fn(usize) -> usize;
}

fn cast_closure_to_fn() {
    // Casting a closure to a function pointer should be permitted.
    let id = |x| x;
    let _ = id as fn(usize) -> usize;
}

fn main() {}
