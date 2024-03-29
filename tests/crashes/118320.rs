//@ known-bug: #118320
#![feature(const_trait_impl, effects, const_closures)]

#[const_trait]
trait Bar {
    fn foo(&self);
}

impl Bar for () {}

const FOO: () = {
    (const || (()).foo())();
};
