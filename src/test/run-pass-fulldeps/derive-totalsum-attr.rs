// aux-build:custom-derive-plugin-attr.rs
// ignore-stage1

#![feature(plugin, rustc_attrs)]
#![plugin(custom_derive_plugin_attr)]

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

#[rustc_derive_TotalSum]
struct Foo {
    seven: Seven,
    bar: Bar,
    baz: isize,
    #[ignore]
    nan: NaN,
}

#[rustc_derive_TotalSum]
struct Bar {
    quux: isize,
    bleh: isize,
    #[ignore]
    nan: NaN2
}

struct NaN;

impl TotalSum for NaN {
    fn total_sum(&self) -> isize {
        panic!();
    }
}

struct NaN2;

pub fn main() {
    let v = Foo {
        seven: Seven,
        bar: Bar {
            quux: 9,
            bleh: 3,
            nan: NaN2
        },
        baz: 80,
        nan: NaN
    };
    assert_eq!(v.total_sum(), 99);
}
