//@[next] compile-flags: -Znext-solver
//@ revisions: next old
//@ edition:2021

#![feature(const_trait_impl, const_closures)]

const trait Bar {
    fn foo(&self);
}

impl Bar for () {
    fn foo(&self) {}
}

const FOO: () = {
    (const || ().foo())();
    //~^ ERROR the trait bound `(): const Bar` is not satisfied
};

fn main() {}
