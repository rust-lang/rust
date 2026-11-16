// Regression test for issue #105227.

//@ run-rustfix

#![allow(unused)]

fn chars0(v :(& str, &str)) -> impl Iterator<Item = char> {
    //~^ HELP add `use<...>` bound and name the elided lifetimes to explicitly capture them
    v.0.chars().chain(v.1.chars())
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bounds
}

fn chars1(v0 : & str, v1 : &str) -> impl Iterator<Item = char> {
    //~^ HELP add `use<...>` bound and name the elided lifetimes to explicitly capture them
    v0.chars().chain(v1.chars())
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bound
}

fn chars2<'b>(v0 : &str, v1 : &'_ str, v2 : &'b str) -> (impl Iterator<Item = char>, &'b str) {
    //~^ HELP add `use<...>` bound and name the elided lifetimes to explicitly capture them
    (v0.chars().chain(v1.chars()), v2)
    //~^ ERROR hidden type for `impl Iterator<Item = char>` captures lifetime that does not appear in bound
}

fn main() {
}
