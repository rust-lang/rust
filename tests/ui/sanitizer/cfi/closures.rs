// Check various forms of dynamic closure calls

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -Z panic-abort-tests -C prefer-dynamic=off
//@ compile-flags: --test
//@ run-pass

#![feature(fn_traits)]
#![feature(unboxed_closures)]

fn foo<'a, T>() -> Box<dyn Fn(&'a T) -> &'a T> {
    Box::new(|x| x)
}

#[test]
fn dyn_fn_with_params() {
    let x = 3;
    let f = foo();
    f(&x);
    // FIXME remove once drops are working.
    std::mem::forget(f);
}

#[test]
fn call_fn_trait() {
   let f: &dyn Fn() = &(|| {}) as _;
   f.call(());
}

#[test]
fn fn_ptr_cast() {
    let f: &fn() = &((|| ()) as _);
    f();
}

fn use_fnmut<F: FnMut()>(mut f: F) {
    f()
}

#[test]
fn fn_to_fnmut() {
    let f: &dyn Fn() = &(|| {}) as _;
    use_fnmut(f);
}

fn hrtb_helper(f: &dyn for<'a> Fn(&'a usize)) {
    f(&10)
}

#[test]
fn hrtb_fn() {
    hrtb_helper((&|x: &usize| println!("{}", *x)) as _)
}

#[test]
fn fnonce() {
    let f: Box<dyn FnOnce()> = Box::new(|| {}) as _;
    f();
}

fn use_closure<C>(call: extern "rust-call" fn(&C, ()) -> i32, f: &C) -> i32 {
    call(f, ())
}

#[test]
fn closure_addr_taken() {
    let x = 3i32;
    let f = || x;
    let call = Fn::<()>::call;
    use_closure(call, &f);
}

fn use_closure_once<C>(call: extern "rust-call" fn(C, ()) -> i32, f: C) -> i32 {
    call(f, ())
}

#[test]
fn closure_once_addr_taken() {
    let g = || 3;
    let call2 = FnOnce::<()>::call_once;
    use_closure_once(call2, g);
}
