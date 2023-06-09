// run-pass
// aux-build:blind-item-mixed-crate-use-item-foo.rs
// aux-build:blind-item-mixed-crate-use-item-foo2.rs

// pretty-expanded FIXME #23616

mod m {
    pub fn f<T>(_: T, _: (), _: ()) { }
    pub fn g<T>(_: T, _: (), _: ()) { }
}

const BAR: () = ();
struct Data;
use m::f;
extern crate blind_item_mixed_crate_use_item_foo as foo;

fn main() {
    const BAR2: () = ();
    struct Data2;
    use m::g;

    extern crate blind_item_mixed_crate_use_item_foo2 as foo2;

    f(Data, BAR, foo::X);
    g(Data2, BAR2, foo2::Y);
}
