// Identifier pattern referring to a const generic parameter is an error (issue #68853).

#![feature(const_generics)] //~ WARN the feature `const_generics` is incomplete

fn check<const N: usize>() {
    match 1 {
        N => {} //~ ERROR const parameters cannot be referenced in patterns
        _ => {}
    }
}

fn main() {}
