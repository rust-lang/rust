#![feature(export)]
#![crate_type = "sdylib"]

// interface file is broken(priv fn):
#[export]
extern "C" fn foo();
