//@ check-pass
#![deny(dead_code)]

enum Foo {
    Bar,
}

fn main() {
    let p = [0; 0];
    p.bar();
}

trait Bar {
    fn bar(&self) -> usize {
        3
    }
}

impl Bar for [u32; Foo::Bar as usize] {}
