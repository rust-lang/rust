// run-pass
// aux-build:derive-no-std.rs

extern crate derive_no_std;
use derive_no_std::*;

fn main() {
    let f = Foo { x: 0 };
    assert_eq!(f.clone(), Foo::default());

    assert!(Bar::Qux < Bar::Quux(42));
}
