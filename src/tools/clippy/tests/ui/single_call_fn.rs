//@ignore-bitwidth: 32
//@aux-build:proc_macros.rs
#![allow(clippy::redundant_closure_call, unused)]
#![warn(clippy::single_call_fn)]
#![no_main]

#[macro_use]
extern crate proc_macros;

// Do not lint since it's public
pub fn f() {}

fn i() {}
//~^ single_call_fn
fn j() {}
//~^ single_call_fn

fn h() {
    // Linted
    let a = i;
    // Do not lint closures
    let a = (|| {
        // Not linted
        a();
        // Imo, it's reasonable to lint this as the function is still only being used once. Just in
        // a closure.
        j();
    });
    a();
}

fn g() {
    f();
}

fn c() {
    //~^ single_call_fn
    println!("really");
    println!("long");
    println!("function...");
}

fn d() {
    c();
}

fn a() {}
//~^ single_call_fn

fn b() {
    a();

    external! {
        fn lol() {
            lol_inner();
        }
        fn lol_inner() {}
    }
    with_span! {
        span
        fn lol2() {
            lol2_inner();
        }
        fn lol2_inner() {}
    }
}

fn e() {
    b();
    b();
}

#[test]
fn k() {}

mod issue12182 {
    #[allow(clippy::single_call_fn)]
    fn print_foo(text: &str) {
        println!("{text}");
    }

    fn use_print_foo() {
        print_foo("foo");
    }
}

#[test]
fn l() {
    k();
}

trait Trait {
    fn default() {}
    //~^ single_call_fn
    fn foo(&self);
}
unsafe extern "C" {
    // test some kind of foreign item
    fn rand() -> std::ffi::c_int;
}
fn m<T: Trait>(v: T) {
    const NOT_A_FUNCTION: i32 = 1;
    let _ = NOT_A_FUNCTION;

    struct S;
    impl S {
        fn foo() {}
        //~^ single_call_fn
    }
    T::default();
    S::foo();
    v.foo();
    unsafe { rand() };
}
