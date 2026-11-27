//@ edition: 2015
//@ compile-flags: --error-format human

pub fn main() {
    let x: Iter;
}

//~? RAW cannot find type `Iter` in this scope
