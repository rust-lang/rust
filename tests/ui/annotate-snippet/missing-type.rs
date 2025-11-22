//@ compile-flags: --error-format human-annotate-rs -Z unstable-options
//@ edition: 2015

pub fn main() {
    let x: Iter;
}

//~? RAW cannot find type `Iter` in this scope
