//! Test that `#[derive(Debug)]` for enums correctly formats variant names.

//@ run-pass

#[derive(Debug)]
enum Foo {
    A(usize),
    C,
}

#[derive(Debug)]
enum Bar {
    D,
}

pub fn main() {
    // Test variant with data
    let foo_a = Foo::A(22);
    assert_eq!("A(22)".to_string(), format!("{:?}", foo_a));

    if let Foo::A(value) = foo_a {
        println!("Value: {}", value); // This needs to remove #[allow(dead_code)]
    }

    // Test unit variant
    assert_eq!("C".to_string(), format!("{:?}", Foo::C));

    // Test unit variant from different enum
    assert_eq!("D".to_string(), format!("{:?}", Bar::D));
}
