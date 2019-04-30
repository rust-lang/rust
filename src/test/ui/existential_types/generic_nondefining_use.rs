#![feature(existential_type)]

fn main() {}

existential type Cmp<T>: 'static;
//~^ ERROR could not find defining uses
//~^^ ERROR: at least one trait must be specified


// not a defining use, because it doesn't define *all* possible generics
fn cmp() -> Cmp<u32> { //~ ERROR defining existential type use does not fully define
    5u32
}
