//@ build-pass
//@ compile-flags: -Clink-dead-code=true

// Regression test for https://github.com/rust-lang/rust/issues/155803

#![feature(const_closures, const_trait_impl)]

const _: () = {
    assert!((const || true)());
};

fn main() {}
