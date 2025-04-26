//@ edition:2015
//@ run-rustfix
//@ compile-flags: -Adead_code

#[derive(Debug)]
struct Symbol;

type Ident = Path;
//~^ ERROR cannot find type `Path` in this scope

fn main() {}
