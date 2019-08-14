// run-pass

// aux-build:specialization_cross_crate.rs

#![feature(specialization)]

extern crate specialization_cross_crate;

use specialization_cross_crate::*;

struct NotClone;

#[derive(Clone)]
struct MarkedAndClone;
impl MyMarker for MarkedAndClone {}

struct MyType<T>(T);
impl<T> Foo for MyType<T> {
    default fn foo(&self) -> &'static str {
        "generic MyType"
    }
}

impl Foo for MyType<u8> {
    fn foo(&self) -> &'static str {
        "MyType<u8>"
    }
}

struct MyOtherType;
impl Foo for MyOtherType {}

fn  main() {
    assert!(NotClone.foo() == "generic");
    assert!(0u8.foo() == "generic Clone");
    assert!(vec![NotClone].foo() == "generic");
    assert!(vec![0u8].foo() == "generic Vec");
    assert!(vec![0i32].foo() == "Vec<i32>");
    assert!(0i32.foo() == "i32");
    assert!(String::new().foo() == "String");
    assert!(((), 0).foo() == "generic pair");
    assert!(((), ()).foo() == "generic uniform pair");
    assert!((0u8, 0u32).foo() == "(u8, u32)");
    assert!((0u8, 0u8).foo() == "(u8, u8)");
    assert!(MarkedAndClone.foo() == "generic Clone + MyMarker");

    assert!(MyType(()).foo() == "generic MyType");
    assert!(MyType(0u8).foo() == "MyType<u8>");
    assert!(MyOtherType.foo() == "generic");
}
