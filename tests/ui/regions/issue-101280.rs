use std::cell::Cell;

type Ty = for<'r> fn(Cell<(&'r i32, &'r i32)>);

fn f<'r>(f: fn(Cell<(&'r i32, &i32)>)) -> Ty {
    f
    //~^ ERROR mismatched types
}

fn main() {}
