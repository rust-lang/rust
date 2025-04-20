#![feature(export)]

// interface file is broken(priv fn):
#[export]
extern "C" fn foo();
