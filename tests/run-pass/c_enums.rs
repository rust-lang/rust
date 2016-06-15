enum Foo {
    Bar = 42,
    Baz,
    Quux = 100,
}

enum Signed {
    Bar = -42,
    Baz,
    Quux = 100,
}

fn foo() -> [u8; 3] {
    [Foo::Bar as u8, Foo::Baz as u8, Foo::Quux as u8]
}

fn signed() -> [i8; 3] {
    [Signed::Bar as i8, Signed::Baz as i8, Signed::Quux as i8]
}

fn unsafe_match() -> bool {
    match unsafe { std::mem::transmute::<u8, Foo>(43) } {
        Foo::Baz => true,
        _ => false,
    }
}

fn main() {
    assert_eq!(foo(), [42, 43, 100]);
    assert_eq!(signed(), [-42, -41, 100]);
    assert!(unsafe_match());
}
