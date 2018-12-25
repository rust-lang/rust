// Verifies that static items can't be moved

struct B;

struct Foo {
    foo: isize,
    b: B,
}

static BAR: Foo = Foo { foo: 5, b: B };


fn test(f: Foo) {
    let _f = Foo{foo: 4, ..f};
}

fn main() {
    test(BAR); //~ ERROR cannot move out of static item
}
