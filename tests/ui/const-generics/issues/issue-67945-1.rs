//@ revisions: full min
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(generic_const_exprs))]

use std::mem::MaybeUninit;

struct Bug<S> {
    //[min]~^ ERROR parameter `S` is never used
    A: [(); {
        let x: S = MaybeUninit::uninit();
        //[min]~^ ERROR generic parameters may not be used in const operations
        //[full]~^^ ERROR mismatched types
        let b = &*(&x as *const _ as *const S);
        //[min]~^ ERROR generic parameters may not be used in const operations
        0
    }],
}

fn main() {}
