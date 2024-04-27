//@ run-pass
struct Foo<A = (isize, char)> {
    a: A
}

impl Foo<isize> {
    fn bar_int(&self) -> isize {
        self.a
    }
}

impl Foo<char> {
    fn bar_char(&self) -> char {
        self.a
    }
}

impl Foo {
    fn bar(&self) {
        let (i, c): (isize, char) = self.a;
        assert_eq!(Foo { a: i }.bar_int(), i);
        assert_eq!(Foo { a: c }.bar_char(), c);
    }
}

impl<A: Clone> Foo<A> {
    fn baz(&self) -> A {
        self.a.clone()
    }
}

fn default_foo(x: Foo) {
    let (i, c): (isize, char) = x.a;
    assert_eq!(i, 1);
    assert_eq!(c, 'a');

    x.bar();
    assert_eq!(x.baz(), (1, 'a'));
}

#[derive(PartialEq, Debug)]
struct BazHelper<T>(T);

#[derive(PartialEq, Debug)]
// Ensure that we can use previous type parameters in defaults.
struct Baz<T, U = BazHelper<T>, V = Option<U>>(T, U, V);

fn main() {
    default_foo(Foo { a: (1, 'a') });

    let x: Baz<bool> = Baz(true, BazHelper(false), Some(BazHelper(true)));
    assert_eq!(x, Baz(true, BazHelper(false), Some(BazHelper(true))));
}
