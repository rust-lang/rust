// Regression test for issue #105227.

// FIXME(precise_capturing): Add rustfix here after dealing w/ elided lifetimes

#![allow(unused)]

fn chars0(v :(& str, &str)) -> impl Iterator<Item = char> {
    //~^ HELP add a `use<...>` bound
    v.0.chars().chain(v.1.chars())
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bounds
}

fn chars1(v0 : & str, v1 : &str) -> impl Iterator<Item = char> {
    //~^ HELP add a `use<...>` bound
    v0.chars().chain(v1.chars())
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bound
}

fn chars2<'b>(v0 : &str, v1 : &'_ str, v2 : &'b str) -> (impl Iterator<Item = char>, &'b str) {
    //~^ HELP add a `use<...>` bound
    (v0.chars().chain(v1.chars()), v2)
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bound
}

fn main() {
}
