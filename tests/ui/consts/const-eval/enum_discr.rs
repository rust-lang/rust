//@ run-pass

enum Foo {
    X = 42,
    Y = Foo::X as isize - 3,
}

enum Bar {
    X,
    Y = Bar::X as isize + 2,
}

enum Boo {
    X = Boo::Y as isize * 2,
    Y = 9,
}

fn main() {
    assert_eq!(Foo::X as isize, 42);
    assert_eq!(Foo::Y as isize, 39);
    assert_eq!(Bar::X as isize, 0);
    assert_eq!(Bar::Y as isize, 2);
    assert_eq!(Boo::X as isize, 18);
    assert_eq!(Boo::Y as isize, 9);
}
