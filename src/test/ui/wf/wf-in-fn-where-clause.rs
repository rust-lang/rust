// Check that we enforce WF conditions also for where clauses in fn items.


#![allow(dead_code)]

trait MustBeCopy<T:Copy> {
}

fn bar<T,U>() //~ ERROR E0277
    where T: MustBeCopy<U>
{
}


fn main() { }
