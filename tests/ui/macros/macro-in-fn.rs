//@ run-pass
#![feature(decl_macro)]

pub fn moo() {
    pub macro ABC() {{}}
}

fn main() {}
