// run-pass

#![feature(main)]

pub fn main() {
    panic!()
}

#[main]
fn foo() {
}
