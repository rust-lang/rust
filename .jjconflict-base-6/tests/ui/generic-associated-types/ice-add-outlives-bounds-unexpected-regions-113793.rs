//! Regression test for <https://github.com/rust-lang/rust/issues/113793>.
//!
//! Using a GAT with a self-referential lifetime bound (`'b: 'b`) in a
//! fully-qualified path with anonymous lifetimes used to ICE.

//@ check-pass

trait Gat {
    type FooArg<'a, 'b: 'b>;
}

impl Gat for () {
    type FooArg<'a, 'b: 'b> = &'a dyn ToString;
}

struct Test;

impl Iterator for Test {
    type Item = Box<dyn Fn(<() as Gat>::FooArg<'_, '_>)>;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {}
