// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

use std::mem::MaybeUninit;

struct Bug<S> {
    //~^ ERROR parameter `S` is never used
    A: [(); {
        let x: S = MaybeUninit::uninit();
        //[min]~^ ERROR generic parameters must not be used inside of non-trivial constant values
        //[full]~^^ ERROR mismatched types
        let b = &*(&x as *const _ as *const S);
        //[min]~^ ERROR generic parameters must not be used inside of non-trivial constant values
        0
    }],
}

fn main() {}
