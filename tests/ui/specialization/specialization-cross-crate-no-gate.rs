//@ run-pass

// Test that specialization works even if only the upstream crate enables it

//@ aux-build:specialization_cross_crate.rs

extern crate specialization_cross_crate;

use specialization_cross_crate::*;

fn  main() {
    assert!(0u8.foo() == "generic Clone");
    assert!(vec![0u8].foo() == "generic Vec");
    assert!(vec![0i32].foo() == "Vec<i32>");
    assert!(0i32.foo() == "i32");
    assert!(String::new().foo() == "String");
    assert!(((), 0).foo() == "generic pair");
    assert!(((), ()).foo() == "generic uniform pair");
    assert!((0u8, 0u32).foo() == "(u8, u32)");
    assert!((0u8, 0u8).foo() == "(u8, u8)");
}
