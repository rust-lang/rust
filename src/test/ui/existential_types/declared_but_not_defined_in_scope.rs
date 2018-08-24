#![feature(existential_type)]

fn main() {}

mod boo {
    // declared in module but not defined inside of it
    pub existential type Boo: ::std::fmt::Debug; //~ ERROR could not find defining uses
}

fn bomp() -> boo::Boo {
    ""
}
