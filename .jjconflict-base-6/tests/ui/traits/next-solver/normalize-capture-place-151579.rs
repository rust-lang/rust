// Regression test for #151579
//@ compile-flags: -Znext-solver=globally
//@ edition:2018

#![deny(rust_2021_incompatible_closure_captures)]

struct Dummy;

trait Trait {
    type Assoc;
}

impl Trait for Dummy {
    type Assoc = (*mut i32,);
}

struct SyncPointer(<Dummy as Trait>::Assoc);
unsafe impl Sync for SyncPointer {}

fn test_assoc_capture(a: SyncPointer) {
    let _ = move || {
        //~^ ERROR: changes to closure capture
        let _x = a.0.0;
    };
}

fn main() {
    let ptr = SyncPointer((std::ptr::null_mut(),));
    test_assoc_capture(ptr);
}
