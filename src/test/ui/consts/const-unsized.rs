use std::fmt::Debug;

const CONST_0: Debug+Sync = *(&0 as &(Debug+Sync));
//~^ ERROR the size for values of type

const CONST_FOO: str = *"foo";
//~^ ERROR the size for values of type

static STATIC_1: Debug+Sync = *(&1 as &(Debug+Sync));
//~^ ERROR the size for values of type

static STATIC_BAR: str = *"bar";
//~^ ERROR the size for values of type

fn main() {
    println!("{:?} {:?} {:?} {:?}", &CONST_0, &CONST_FOO, &STATIC_1, &STATIC_BAR);
}
