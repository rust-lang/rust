// check-pass
#![feature(c_variadic)]
#![allow(dead_code)]

fn foo() -> u32 { 42 }
fn bar(x: u32) -> u32 { x }
fn baz(x: u32, y: u32) -> u32 { x + y }
unsafe fn unsafe_fn() { }
extern "C" fn c_fn() { }
unsafe extern "C" fn unsafe_c_fn() { }
unsafe extern fn variadic_fn(_x: u32, _args: ...) { }
fn call_fn(f: &dyn Fn(u32) -> u32, x: u32) { f(x); }
fn parameterized_call_fn<F: Fn(u32) -> u32>(f: &F, x: u32) { f(x); }

fn main() {
    let _zst_ref = &foo;
    //~^ WARN cast `foo` with `as fn() -> _` to use it as a pointer
    let fn_item = foo;
    let _indirect_ref = &fn_item;
    //~^ WARN cast `fn_item` with `as fn() -> _` to use it as a pointer
    let _cast_zst_ptr = &foo as *const _;
    //~^ WARN cast `foo` with `as fn() -> _` to use it as a pointer
    let _coerced_zst_ptr: *const _ = &foo;
    //~^ WARN cast `foo` with `as fn() -> _` to use it as a pointer

    let _zst_ref = &mut foo;
    //~^ WARN cast `foo` with `as fn() -> _` to use it as a pointer
    let mut mut_fn_item = foo;
    let _indirect_ref = &mut mut_fn_item;
    //~^ WARN cast `fn_item` with `as fn() -> _` to use it as a pointer
    let _cast_zst_ptr = &mut foo as *mut _;
    //~^ WARN cast `foo` with `as fn() -> _` to use it as a pointer
    let _coerced_zst_ptr: *mut _ = &mut foo;
    //~^ WARN cast `foo` with `as fn() -> _` to use it as a pointer

    let _cast_zst_ref = &foo as &dyn Fn() -> u32;
    let _coerced_zst_ref: &dyn Fn() -> u32 = &foo;

    let _cast_zst_ref = &mut foo as &mut dyn Fn() -> u32;
    let _coerced_zst_ref: &mut dyn Fn() -> u32 = &mut foo;
    let _fn_ptr = foo as fn() -> u32;

    println!("{:p}", &foo);
    //~^ WARN cast `foo` with as fn() -> _` to use it as a pointer
    println!("{:p}", &bar);
    //~^ WARN cast `bar` with as fn(_) -> _` to use it as a pointer
    println!("{:p}", &baz);
    //~^ WARN cast `baz` with as fn(_, _) -> _` to use it as a pointer
    println!("{:p}", &unsafe_fn);
    //~^ WARN cast `baz` with as unsafe fn()` to use it as a pointer
    println!("{:p}", &c_fn);
    //~^ WARN cast `baz` with as extern "C" fn()` to use it as a pointer
    println!("{:p}", &unsafe_c_fn);
    //~^ WARN cast `baz` with as unsafe extern "C" fn()` to use it as a pointer
    println!("{:p}", &variadic_fn);
    //~^ WARN cast `baz` with as unsafe extern "C" fn(_, ...) -> _` to use it as a pointer
    println!("{:p}", &std::env::var::<String>);
    //~^ WARN cast `std::env::var` with as fn(_) -> _` to use it as a pointer

    println!("{:p}", foo as fn() -> u32);

    unsafe {
        std::mem::transmute::<_, usize>(&foo);
        //~^ WARN cast `foo` with as fn() -> _` to use it as a pointer
        std::mem::transmute::<_, usize>(foo as fn() -> u32);
    }

    (&bar)(1);
    call_fn(&bar, 1);
    parameterized_call_fn(&bar, 1);
}
