// run-pass

#![feature(decl_macro)]

pub struct Foo {
    bar: u32,
}
pub macro pattern($a:pat) {
    Foo { bar: $a }
}

fn main() {
    match (Foo { bar: 3 }) {
        pattern!(3) => println!("Test OK"),
        _ => unreachable!(),
    }
}
