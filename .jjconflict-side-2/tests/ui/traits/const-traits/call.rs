//@ check-pass
//@[next] compile-flags: -Znext-solver
//@revisions: next old
#![feature(const_closures, const_trait_impl)]

const _: () = {
    assert!((const || true)());
};

fn main() {}
