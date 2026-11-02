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

#[derive(Debug)]
enum Large {
    A,
    Bar,
    C,
    Distance,
    Emily,
    F,
    G,
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

    // Test large c-like enum, those use a different implementation.
    assert_eq!("A".to_string(), format!("{:?}", Large::A));
    assert_eq!("Bar".to_string(), format!("{:?}", Large::Bar));
    assert_eq!("C".to_string(), format!("{:?}", Large::C));
    assert_eq!("Distance".to_string(), format!("{:?}", Large::Distance));
    assert_eq!("Emily".to_string(), format!("{:?}", Large::Emily));
    assert_eq!("F".to_string(), format!("{:?}", Large::F));
    assert_eq!("G".to_string(), format!("{:?}", Large::G));
}
