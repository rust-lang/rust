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
    //~^ ERROR: casting function pointer `foo` to `i8`
    //~| NOTE: `-D clippy::fn-to-numeric-cast-any` implied by `-D warnings`
    let _ = foo as i16;
    //~^ ERROR: casting function pointer `foo` to `i16`
    let _ = foo as i32;
    //~^ ERROR: casting function pointer `foo` to `i32`
    let _ = foo as i64;
    //~^ ERROR: casting function pointer `foo` to `i64`
    let _ = foo as i128;
    //~^ ERROR: casting function pointer `foo` to `i128`
    let _ = foo as isize;
    //~^ ERROR: casting function pointer `foo` to `isize`

    let _ = foo as u8;
    //~^ ERROR: casting function pointer `foo` to `u8`
    let _ = foo as u16;
    //~^ ERROR: casting function pointer `foo` to `u16`
    let _ = foo as u32;
    //~^ ERROR: casting function pointer `foo` to `u32`
    let _ = foo as u64;
    //~^ ERROR: casting function pointer `foo` to `u64`
    let _ = foo as u128;
    //~^ ERROR: casting function pointer `foo` to `u128`
    let _ = foo as usize;
    //~^ ERROR: casting function pointer `foo` to `usize`
}

fn static_method_to_integer() {
    let _ = Struct::static_method as usize;
    //~^ ERROR: casting function pointer `Struct::static_method` to `usize`
}

fn fn_with_fn_arg(f: fn(i32) -> u32) -> usize {
    f as usize
    //~^ ERROR: casting function pointer `f` to `usize`
}

fn fn_with_generic_static_trait_method<T: Trait>() -> usize {
    T::static_method as usize
    //~^ ERROR: casting function pointer `T::static_method` to `usize`
}

fn closure_to_fn_to_integer() {
    let clos = |x| x * 2_u32;

    let _ = (clos as fn(u32) -> u32) as usize;
    //~^ ERROR: casting function pointer `(clos as fn(u32) -> u32)` to `usize`
}

fn fn_to_raw_ptr() {
    let _ = foo as *const ();
    //~^ ERROR: casting function pointer `foo` to `*const ()`
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
