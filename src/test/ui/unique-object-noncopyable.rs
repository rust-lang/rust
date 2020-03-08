// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
#![feature(box_syntax)]

trait Foo {
    fn f(&self);
}

struct Bar {
    x: isize,
}

impl Drop for Bar {
    fn drop(&mut self) {}
}

impl Foo for Bar {
    fn f(&self) {
        println!("hi");
    }
}

fn main() {
    let x = box Bar { x: 10 };
    let y: Box<dyn Foo> = x as Box<dyn Foo>;
    let _z = y.clone(); //~ ERROR no method named `clone` found
}
