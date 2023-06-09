// run-rustfix
#![deny(rust_2021_incompatible_closure_captures)]
//~^ NOTE: the lint level is defined here

use std::thread;

#[derive(Debug)]
struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

/* Test Send Trait Migration */
struct SendPointer(*mut i32);
unsafe impl Send for SendPointer {}

fn test_send_trait() {
    let mut f = 10;
    let fptr = SendPointer(&mut f as *mut i32);
    thread::spawn(move || unsafe {
        //~^ ERROR: changes to closure capture
        //~| NOTE: in Rust 2018, this closure implements `Send`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `fptr` to be fully captured
        *fptr.0 = 20;
        //~^ NOTE: in Rust 2018, this closure captures all of `fptr`, but in Rust 2021, it will only capture `fptr.0`
    }).join().unwrap();
}

/* Test Sync Trait Migration */
struct CustomInt(*mut i32);
struct SyncPointer(CustomInt);
unsafe impl Sync for SyncPointer {}
unsafe impl Send for CustomInt {}

fn test_sync_trait() {
    let mut f = 10;
    let f = CustomInt(&mut f as *mut i32);
    let fptr = SyncPointer(f);
    thread::spawn(move || unsafe {
        //~^ ERROR: changes to closure capture
        //~| NOTE: in Rust 2018, this closure implements `Sync`
        //~| NOTE: in Rust 2018, this closure implements `Send`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `fptr` to be fully captured
        *fptr.0.0 = 20;
        //~^ NOTE: in Rust 2018, this closure captures all of `fptr`, but in Rust 2021, it will only capture `fptr.0.0`
    }).join().unwrap();
}

/* Test Clone Trait Migration */
struct S(Foo);
struct T(i32);

struct U(S, T);

impl Clone for U {
    fn clone(&self) -> Self {
        U(S(Foo(0)), T(0))
    }
}

fn test_clone_trait() {
    let f = U(S(Foo(0)), T(0));
    let c = || {
        //~^ ERROR: changes to closure capture in Rust 2021 will affect drop order and which traits the closure implements
        //~| NOTE: in Rust 2018, this closure implements `Clone`
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `f` to be fully captured
        let f_1 = f.1;
        //~^ NOTE: in Rust 2018, this closure captures all of `f`, but in Rust 2021, it will only capture `f.1`
        println!("{:?}", f_1.0);
    };

    let c_clone = c.clone();

    c_clone();
}
//~^ NOTE: in Rust 2018, `f` is dropped here, but in Rust 2021, only `f.1` will be dropped here as part of the closure

fn main() {
    test_send_trait();
    test_sync_trait();
    test_clone_trait();
}
