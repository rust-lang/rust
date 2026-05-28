//@ check-pass

fn main() {}

const fn foo() {
    let x = [1, 2, 3, 4, 5];
    let y: &[_] = &x;

    struct Foo<T: ?Sized>(bool, T);

    let x: Foo<[u8; 3]> = Foo(true, [1, 2, 3]);
    let y: &Foo<[u8]> = &x;
}
