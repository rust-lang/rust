// check-pass

#![feature(const_closures, const_trait_impl)]
#![allow(incomplete_features)]

pub const _: () = {
    assert!((const || true)());
};

fn main() {}
