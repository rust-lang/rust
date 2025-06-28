//@ aux-build:staged-api.rs

// Ensure that we enforce const stability of traits in `[const]`/`const` bounds.

#![feature(const_trait_impl)]

use std::ops::Deref;

extern crate staged_api;
use staged_api::MyTrait;

#[const_trait]
trait Foo: [const] MyTrait {
    //~^ ERROR use of unstable const library feature `unstable`
    type Item: [const] MyTrait;
    //~^ ERROR use of unstable const library feature `unstable`
}

const fn where_clause<T>() where T: [const] MyTrait {}
//~^ ERROR use of unstable const library feature `unstable`

const fn nested<T>() where T: Deref<Target: [const] MyTrait> {}
//~^ ERROR use of unstable const library feature `unstable`

const fn rpit() -> impl [const] MyTrait { Local }
//~^ ERROR use of unstable const library feature `unstable`

struct Local;
impl const MyTrait for Local {
//~^ ERROR use of unstable const library feature `unstable`
    fn func() {}
}

fn main() {}
