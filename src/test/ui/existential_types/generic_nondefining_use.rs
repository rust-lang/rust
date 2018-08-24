#![feature(existential_type)]

fn main() {}

existential type Cmp<T>: 'static;

// not a defining use, because it doesn't define *all* possible generics
fn cmp() -> Cmp<u32> { //~ ERROR non-defining existential type use in defining scope
    5u32
}
