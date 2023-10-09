// On unresolved imports, do not suggest hygienic names from different syntax contexts.
#![feature(decl_macro)]

mod module {
    macro make() {
        pub struct Struct;
    }

    make!();
}

// Do not suggest `Struct` since isn't accessible in this syntax context.
use module::Strukt; //~ ERROR unresolved import `module::Strukt`

pub struct Type;

environment!();

macro environment() {
    // Just making sure that the implementation has correctly adjusted the
    // spans to the right expansion and keeps suggesting `Type` here.
    use crate::Typ;
    //~^ ERROR unresolved import `crate::Typ`
    //~| HELP a similar name exists in the module
}

fn main() {}
