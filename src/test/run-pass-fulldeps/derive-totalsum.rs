// aux-build:custom-derive-plugin.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(custom_derive_plugin)]

trait TotalSum {
    fn total_sum(&self) -> isize;
}

impl TotalSum for isize {
    fn total_sum(&self) -> isize {
        *self
    }
}

struct Seven;

impl TotalSum for Seven {
    fn total_sum(&self) -> isize {
        7
    }
}

#[derive_TotalSum]
struct Foo {
    seven: Seven,
    bar: Bar,
    baz: isize,
}

#[derive_TotalSum]
struct Bar {
    quux: isize,
    bleh: isize,
}


pub fn main() {
    let v = Foo {
        seven: Seven,
        bar: Bar {
            quux: 9,
            bleh: 3,
        },
        baz: 80,
    };
    assert_eq!(v.total_sum(), 99);
}
