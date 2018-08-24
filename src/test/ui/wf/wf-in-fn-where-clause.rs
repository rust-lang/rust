// Check that we enforce WF conditions also for where clauses in fn items.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait MustBeCopy<T:Copy> {
}

fn bar<T,U>() //~ ERROR E0277
    where T: MustBeCopy<U>
{
}

#[rustc_error]
fn main() { }
