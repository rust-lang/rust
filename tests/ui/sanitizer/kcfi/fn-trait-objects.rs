// Verifies that types that implement the Fn, FnMut, or FnOnce traits can be
// called through their trait methods.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Cpanic=abort -Cprefer-dynamic=off -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=kcfi  -Zpanic_abort_tests --test
//@ run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

fn foo(_a: u32) {}

#[test]
fn test_fn_trait() {
    let f: Box<dyn Fn(u32)> = Box::new(foo);
    Fn::call(&f, (0,));
}

#[test]
fn test_fnmut_trait() {
    let mut a = 0;
    let mut f: Box<dyn FnMut(u32)> = Box::new(|x| a += x);
    FnMut::call_mut(&mut f, (1,));
}

#[test]
fn test_fnonce_trait() {
    let f: Box<dyn FnOnce(u32)> = Box::new(foo);
    FnOnce::call_once(f, (2,));
}
