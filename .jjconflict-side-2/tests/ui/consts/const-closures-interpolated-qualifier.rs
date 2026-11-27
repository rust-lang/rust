// Ensure that we can still recognize const closures in the parser even if some of the other closure
// qualifiers were interpolated.

//@ check-pass
#![feature(const_closures, const_destruct, const_trait_impl)]

use std::marker::Destruct;

macro_rules! make {
    ($qual:ident $local:ident) => {
        const $qual || { let _ = $local.len(); }
    }
}

const fn scope() {
    let local = String::new();
    call(make!(move local))
}

const fn call(_: impl [const] FnOnce() + [const] Destruct + 'static) {}

fn main() {}
