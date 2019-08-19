// run-pass

#![feature(specialization)]

// Tests a variety of basic specialization scenarios and method
// dispatch for them.

trait Foo {
    fn foo(&self) -> &'static str;
}

impl<T> Foo for T {
    default fn foo(&self) -> &'static str {
        "generic"
    }
}

impl<T: Clone> Foo for T {
    default fn foo(&self) -> &'static str {
        "generic Clone"
    }
}

impl<T, U> Foo for (T, U) where T: Clone, U: Clone {
    default fn foo(&self) -> &'static str {
        "generic pair"
    }
}

impl<T: Clone> Foo for (T, T) {
    default fn foo(&self) -> &'static str {
        "generic uniform pair"
    }
}

impl Foo for (u8, u32) {
    default fn foo(&self) -> &'static str {
        "(u8, u32)"
    }
}

impl Foo for (u8, u8) {
    default fn foo(&self) -> &'static str {
        "(u8, u8)"
    }
}

impl<T: Clone> Foo for Vec<T> {
    default fn foo(&self) -> &'static str {
        "generic Vec"
    }
}

impl Foo for Vec<i32> {
    fn foo(&self) -> &'static str {
        "Vec<i32>"
    }
}

impl Foo for String {
    fn foo(&self) -> &'static str {
        "String"
    }
}

impl Foo for i32 {
    fn foo(&self) -> &'static str {
        "i32"
    }
}

struct NotClone;

trait MyMarker {}
impl<T: Clone + MyMarker> Foo for T {
    default fn foo(&self) -> &'static str {
        "generic Clone + MyMarker"
    }
}

#[derive(Clone)]
struct MarkedAndClone;
impl MyMarker for MarkedAndClone {}

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
}
