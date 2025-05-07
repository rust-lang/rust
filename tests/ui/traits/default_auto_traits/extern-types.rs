//@  compile-flags:  -Zexperimental-default-bounds
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver

#![feature(auto_traits, extern_types, lang_items, negative_impls, no_core, rustc_attrs)]
#![allow(incomplete_features)]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
pub trait Copy {}

#[lang = "default_trait1"]
auto trait Leak {}

// implicit T: Leak here
fn foo<T: ?Sized>(_: &T) {}

mod extern_leak {
    use crate::*;

    extern "C" {
        type Opaque;
    }

    fn forward_extern_ty(x: &Opaque) {
        // ok, extern type leak by default
        crate::foo(x);
    }
}

mod extern_non_leak {
    use crate::*;

    extern "C" {
        type Opaque;
    }

    impl !Leak for Opaque {}
    fn forward_extern_ty(x: &Opaque) {
        foo(x);
        //~^ ERROR: the trait bound `extern_non_leak::Opaque: Leak` is not satisfied
    }
}

fn main() {}
