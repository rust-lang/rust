//@ run-rustfix
#![deny(rust_2021_incompatible_closure_captures)]
//~^ NOTE: the lint level is defined here

use std::thread;

#[derive(Debug)]
struct Foo(String);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

impl Foo {
    fn from(s: &str) -> Self {
        Self(String::from(s))
    }
}

struct S(#[allow(dead_code)] Foo);

#[derive(Clone)]
struct T(#[allow(dead_code)] i32);

struct U(S, T);

impl Clone for U {
    fn clone(&self) -> Self {
        U(S(Foo::from("Hello World")), T(0))
    }
}

fn test_multi_issues() {
    let f1 = U(S(Foo::from("foo")), T(0));
    let f2 = U(S(Foo::from("bar")), T(0));
    let c = || {
        //~^ ERROR: changes to closure capture in Rust 2021
        //~| NOTE: in Rust 2018, this closure implements `Clone` as `f1` implements `Clone`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `f1`, `f2` to be fully captured
        let _f_1 = f1.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `f1`, but in Rust 2021, it will only capture `f1.0`
        let _f_2 = f2.1;
        //~^ NOTE: in Rust 2018, this closure captures all of `f2`, but in Rust 2021, it will only capture `f2.1`
    };

    let c_clone = c.clone();

    c_clone();
}
//~^ NOTE: in Rust 2018, `f2` is dropped here, but in Rust 2021, only `f2.1` will be dropped here as part of the closure

fn test_capturing_all_disjoint_fields_individually() {
    let f1 = U(S(Foo::from("foo")), T(0));
    let c = || {
        //~^ ERROR: changes to closure capture in Rust 2021 will affect which traits the closure implements [rust_2021_incompatible_closure_captures]
        //~| NOTE: in Rust 2018, this closure implements `Clone` as `f1` implements `Clone`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `f1` to be fully captured
        let _f_1 = f1.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `f1`, but in Rust 2021, it will only capture `f1.0`
        let _f_2 = f1.1;
    };

    let c_clone = c.clone();

    c_clone();
}

struct U1(S, T, S);

impl Clone for U1 {
    fn clone(&self) -> Self {
        U1(S(Foo::from("foo")), T(0), S(Foo::from("bar")))
    }
}

fn test_capturing_several_disjoint_fields_individually_1() {
    let f1 = U1(S(Foo::from("foo")), T(0), S(Foo::from("bar")));
    let c = || {
        //~^ ERROR: changes to closure capture in Rust 2021 will affect which traits the closure implements [rust_2021_incompatible_closure_captures]
        //~| NOTE: in Rust 2018, this closure implements `Clone` as `f1` implements `Clone`
        //~| NOTE: in Rust 2018, this closure implements `Clone` as `f1` implements `Clone`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `f1` to be fully captured
        let _f_0 = f1.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `f1`, but in Rust 2021, it will only capture `f1.0`
        let _f_2 = f1.2;
        //~^ NOTE: in Rust 2018, this closure captures all of `f1`, but in Rust 2021, it will only capture `f1.2`
    };

    let c_clone = c.clone();

    c_clone();
}

fn test_capturing_several_disjoint_fields_individually_2() {
    let f1 = U1(S(Foo::from("foo")), T(0), S(Foo::from("bar")));
    let c = || {
        //~^ ERROR: changes to closure capture in Rust 2021 will affect drop order and which traits the closure implements
        //~| NOTE: in Rust 2018, this closure implements `Clone` as `f1` implements `Clone`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `f1` to be fully captured
        let _f_0 = f1.0;
        //~^ NOTE: in Rust 2018, this closure captures all of `f1`, but in Rust 2021, it will only capture `f1.0`
        let _f_1 = f1.1;
        //~^ NOTE: in Rust 2018, this closure captures all of `f1`, but in Rust 2021, it will only capture `f1.1`
    };

    let c_clone = c.clone();

    c_clone();
}
//~^ NOTE: in Rust 2018, `f1` is dropped here, but in Rust 2021, only `f1.1` will be dropped here as part of the closure
//~| NOTE: in Rust 2018, `f1` is dropped here, but in Rust 2021, only `f1.0` will be dropped here as part of the closure

struct SendPointer(*mut i32);
unsafe impl Send for SendPointer {}

struct CustomInt(*mut i32);
struct SyncPointer(CustomInt);
unsafe impl Sync for SyncPointer {}
unsafe impl Send for CustomInt {}

fn test_multi_traits_issues() {
    let mut f1 = 10;
    let f1 = CustomInt(&mut f1 as *mut i32);
    let fptr1 = SyncPointer(f1);

    let mut f2 = 10;
    let fptr2 = SendPointer(&mut f2 as *mut i32);
    thread::spawn(move || unsafe {
        //~^ ERROR: changes to closure capture in Rust 2021
        //~| NOTE: in Rust 2018, this closure implements `Sync` as `fptr1` implements `Sync`
        //~| NOTE: in Rust 2018, this closure implements `Send` as `fptr1` implements `Send`
        //~| NOTE: in Rust 2018, this closure implements `Send` as `fptr2` implements `Send`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `fptr1`, `fptr2` to be fully captured
        *fptr1.0.0 = 20;
        //~^ NOTE: in Rust 2018, this closure captures all of `fptr1`, but in Rust 2021, it will only capture `fptr1.0.0`
        *fptr2.0 = 20;
        //~^ NOTE: in Rust 2018, this closure captures all of `fptr2`, but in Rust 2021, it will only capture `fptr2.0`
    }).join().unwrap();
}

fn main() {
    test_multi_issues();
    test_capturing_all_disjoint_fields_individually();
    test_capturing_several_disjoint_fields_individually_1();
    test_capturing_several_disjoint_fields_individually_2();
    test_multi_traits_issues();
}
