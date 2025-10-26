use std::fmt::Debug;

const CONST_0: dyn Debug + Sync = *(&0 as &(dyn Debug + Sync));
//~^ ERROR the size for values of type
//~| ERROR cannot move out of a shared reference

const CONST_FOO: str = *"foo";
//~^ ERROR the size for values of type
//~| ERROR cannot move out of a shared reference

static STATIC_1: dyn Debug + Sync = *(&1 as &(dyn Debug + Sync));
//~^ ERROR the size for values of type

static STATIC_BAR: str = *"bar";
//~^ ERROR the size for values of type

fn main() {
    println!("{:?} {:?} {:?} {:?}", &CONST_0, &CONST_FOO, &STATIC_1, &STATIC_BAR);
}
