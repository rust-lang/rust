// run-pass

#![allow(warnings, unused)]

enum Foo {
    Variant1(u32, String),
    // TODO(zhamlin): fix issue with projection and multi variants
    // Variant2(u32),
}

fn main() {
    let f: Foo::Variant1 = Foo::Variant1(3, "test".to_string());
    assert_eq!(f.0, 3);
    assert_eq!(f.1, "test".to_string());
}

