//! `#![feature(unsized_fn_params)]` lets you use unsized function parameters. In particular this
//! is load bearing for `Box<dyn FnOnce()>: FnOnce()`. To do that, borrowck relaxes the requirement
//! that certain places must be `Sized`. But in #142911 we removed alloca support, so these
//! arguments cannot be put in temporaries (or ICE at codegen) That means when `unsized_fn_params`
//! is enabled, we must explicitly check that unsized function arguments are place expressions.
//!
//! Also see tests/ui/unsized_locals/unsized-exprs-rpass.rs

#![feature(unsized_fn_params)]

fn foo() -> Box<[u8]> {
    Box::new(*b"foo")
}

fn udrop<T: ?Sized>(_x: T) {}

fn main(){
    // NB The ordering of the following operations matters, otherwise errors get swallowed somehow.

    udrop::<[u8]>(if true { *foo() } else { *foo() }); //~ERROR the size for values of type `[u8]` cannot be known at compilation time
    udrop::<[u8]>({ *foo() }); //~ERROR the size for values of type `[u8]` cannot be known at compilation time
    udrop(match foo() { x => *x }); //~ERROR the size for values of type `[u8]` cannot be known at compilation time
    udrop::<[u8]>({ loop { break *foo(); } }); //~ERROR the size for values of type `[u8]` cannot be known at compilation time

    { *foo() }; //~ERROR the size for values of type `[u8]` cannot be known at compilation time
    { loop { break *foo(); } }; //~ERROR the size for values of type `[u8]` cannot be known at compilation time
}
