#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

// Regression test for #52050: when inserting the blanket impl `I`
// into the tree, we had to replace the child node for `Foo`, which
// led to the structure of the tree being messed up.

use std::iter::Iterator;

trait IntoPyDictPointer { }

struct Foo { }

impl Iterator for Foo {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        None
    }
}

impl IntoPyDictPointer for Foo { }

impl<I> IntoPyDictPointer for I
where
    I: Iterator,
{
}

impl IntoPyDictPointer for () //~ ERROR conflicting implementations
{
}

fn main() { }
