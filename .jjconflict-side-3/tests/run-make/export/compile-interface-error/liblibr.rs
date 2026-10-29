#![feature(export_stable)]

// interface file is broken(priv fn):
#[export_stable]
extern "C" fn foo();
