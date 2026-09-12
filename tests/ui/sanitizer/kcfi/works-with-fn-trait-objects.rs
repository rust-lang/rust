// Verifies that types that implement the Fn, FnMut, or FnOnce traits can be
// called through their trait methods.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

fn foo1(x: i32) -> i32 {
    x
}

fn main() {
    // Types that implement Fn
    // The <fn(i32) -> i32 as Fn<(i32,)>>::call FnPtrShim in the vtable is transformed into
    // <dyn Fn(i32) -> i32 as Fn<(i32,)>>::call.
    let f: Box<dyn Fn(i32) -> i32> = Box::new(foo1);
    // The virtual method call is transformed into <dyn Fn(i32) -> i32 as Fn<(i32,)>>::call
    assert_eq!(Fn::call(&f, (1,)), 1);

    // Types that implement FnMut
    let mut a = 0;
    // The closure is transformed into <dyn FnMut(i32) as FnMut<(i32,)>>::call_mut
    let mut f: Box<dyn FnMut(i32)> = Box::new(|x| a += x);
    // The virtual method call is transformed into <dyn FnMut(i32) as FnMut<(i32,)>>::call_mut
    FnMut::call_mut(&mut f, (2,));
    drop(f);
    assert_eq!(a, 2);

    // Types that implement FnOnce
    // The <fn(i32) -> i32 as FnOnce<(i32,)>>::call_once FnPtrShim in the vtable is transformed
    // into <dyn FnOnce(i32) -> i32 as FnOnce<(i32,)>>::call_once.
    let f: Box<dyn FnOnce(i32) -> i32> = Box::new(foo1);
    // The virtual method call is transformed into
    // <dyn FnOnce(i32) -> i32 as FnOnce<(i32,)>>::call_once.
    assert_eq!(FnOnce::call_once(f, (3,)), 3);
}
