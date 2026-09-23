//@ check-fail
//@ run-rustfix

#![no_std]
#![crate_type = "lib"]
#![allow(dead_code)]

struct S;

fn drop_md() {
    core::mem::drop(core::mem::ManuallyDrop::new(S));
    //~^ ERROR calls to `core::mem::drop`
}
