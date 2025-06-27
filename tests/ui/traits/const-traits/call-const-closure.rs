//@ compile-flags: -Znext-solver
//@ edition:2021

#![feature(const_trait_impl, const_closures)]
#![allow(incomplete_features)]

#[const_trait]
trait Bar {
    fn foo(&self);
}

impl Bar for () {
    fn foo(&self) {}
}

const FOO: () = {
    (const || ().foo())();
    //~^ ERROR the trait bound `(): [const] Bar` is not satisfied
    // FIXME(const_trait_impl): The constness environment for const closures is wrong.
};

fn main() {}
