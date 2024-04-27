// Identifier pattern referring to an ambiguity item is an error (issue #46079).

mod m {
    pub fn f() {}
}
use m::*;

mod n {
    pub fn f() {}
}
use n::*; // OK, no conflict with `use m::*;`

fn main() {
    let v = f; //~ ERROR `f` is ambiguous
    match v {
        f => {} //~ ERROR `f` is ambiguous
        mut f => {} // OK, unambiguously a fresh binding due to `mut`
    }
}
