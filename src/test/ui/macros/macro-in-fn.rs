#![feature(decl_macro)]

pub fn moo() {
    pub macro ABC() {{}}
}

fn main() {
    ABC!(); //~ ERROR cannot find macro `ABC!` in this scope
}
