//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Check that we test WF conditions for fn where clauses in a trait definition.

#![allow(dead_code)]
#![allow(unused_variables)]

struct Bar<T: Eq + ?Sized> {
    value: Box<T>,
}

trait Foo {
    fn bar(&self)
    where
        Self: Sized,
        Bar<Self>: Copy;
    //~^ ERROR E0277
    //
    // Here, Eq ought to be implemented.
}

fn main() {}
