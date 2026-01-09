//@ edition:2015..2021
//@ revisions: classic relocate
//@ [classic] compile-flags: -Z pack-coroutine-layout=no
//@ [relocate] compile-flags: -Z pack-coroutine-layout=captures-only
//@ compile-flags: -Z print-type-sizes --crate-type=lib
//@ build-pass
//@ ignore-pass

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

fn coroutine<const C: usize>(array: [u8; C]) -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine]
    move |()| {
        yield ();
        let _ = array;
    }
}

pub fn foo() {
    let _ = coroutine([0; 8192]);
}
