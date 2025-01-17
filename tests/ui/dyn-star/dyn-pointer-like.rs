// Test that `dyn PointerLike` and `dyn* PointerLike` do not implement `PointerLike`.
// This used to ICE during codegen.

#![crate_type = "lib"]

#![feature(pointer_like_trait, dyn_star)]
#![feature(unsized_fn_params)]
#![expect(incomplete_features)]
#![expect(internal_features)]

use std::marker::PointerLike;

pub fn lol(x: dyn* PointerLike) {
    foo(x); //~ ERROR `dyn* PointerLike` needs to have the same ABI as a pointer
}

pub fn uwu(x: dyn PointerLike) {
    foo(x); //~ ERROR `dyn PointerLike` needs to have the same ABI as a pointer
}

fn foo<T: PointerLike + ?Sized>(x: T) {
    let _: dyn* PointerLike = x;
}
