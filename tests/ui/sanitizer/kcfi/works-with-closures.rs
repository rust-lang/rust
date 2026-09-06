// Verifies that closures can be called through various forms of dynamic calls
// (i.e., through trait objects of the Fn, FnMut, and FnOnce traits, and as
// function pointers).
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

fn foo1<'a, T>() -> Box<dyn Fn(&'a T) -> &'a T> {
    // The closure is transformed into <dyn Fn(&T) -> &T as Fn<(&T,)>>::call
    Box::new(|x| x)
}

fn use_fnmut<F: FnMut() -> i32>(mut f: F) -> i32 {
    // The virtual method call is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    f()
}

fn use_closure<C>(call: extern "rust-call" fn(&C, ()) -> i32, f: &C) -> i32 {
    // The indirect call is not transformed, as the type id is encoded from the fn pointer type
    call(f, ())
}

fn use_closure_once<C>(call: extern "rust-call" fn(C, ()) -> i32, f: C) -> i32 {
    // The indirect call is not transformed, as the type id is encoded from the fn pointer type
    call(f, ())
}

fn main() {
    // Closures with parameters, through a dyn Fn trait object
    let x = 1;
    let f = foo1();
    // The virtual method call is transformed into <dyn Fn(&T) -> &T as Fn<(&T,)>>::call
    assert_eq!(*f(&x), 1);

    // Closures, through the Fn trait method
    // The closure is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    let f: &dyn Fn() -> i32 = &(|| 2) as _;
    // The virtual method call is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    assert_eq!(f.call(()), 2);

    // Fn closures passed where FnMut is expected
    // The closure is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    let f: &dyn Fn() -> i32 = &(|| 3) as _;
    assert_eq!(use_fnmut(f), 3);

    // FnOnce closures, through a dyn FnOnce trait object
    // <dyn FnOnce() -> i32 as FnOnce<()>>::call_once receives an unsizeable `self: Self`, so the
    // VTableShim for it in the vtable is transformed into
    // <dyn FnOnce() -> i32 as FnOnce<()>>::call_once.
    let f: Box<dyn FnOnce() -> i32> = Box::new(|| 4) as _;
    // The virtual method call is transformed into <dyn FnOnce() -> i32 as FnOnce<()>>::call_once
    assert_eq!(f(), 4);

    // Closures that move out of a capture, and so are FnOnce and not Fn or FnMut
    let x = Box::new(5);
    // The closure is transformed into <dyn FnOnce() -> i32 as FnOnce<()>>::call_once
    let f: Box<dyn FnOnce() -> i32> = Box::new(move || {
        drop(x);
        5
    });
    assert_eq!(f(), 5);

    // Closures cast to function pointers
    // The closure is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    let f: fn() -> i32 = std::hint::black_box(|| 6);
    // The indirect call is not transformed, as the type id is encoded from the fn pointer type
    assert_eq!(f(), 6);

    // Closures with Fn::call cast to function pointers
    let x = 7;
    // The closure is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    let f = || x;
    let call = std::hint::black_box(Fn::<()>::call);
    assert_eq!(use_closure(call, &f), 7);

    // Closures with FnOnce::call_once cast to function pointers
    // The closure is transformed into <dyn Fn() -> i32 as Fn<()>>::call
    let g = || 8;
    // The ClosureOnceShim is not transformed, as it can not be called through a vtable
    let call = std::hint::black_box(FnOnce::<()>::call_once);
    assert_eq!(use_closure_once(call, g), 8);
}
