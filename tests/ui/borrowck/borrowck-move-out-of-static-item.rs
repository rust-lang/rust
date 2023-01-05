// Ensure that moves out of static items is forbidden

struct Foo {
    foo: isize,
}

static BAR: Foo = Foo { foo: 5 };


fn test(f: Foo) {
    let _f = Foo{foo: 4, ..f};
}

fn main() {
    test(BAR); //~ ERROR cannot move out of static item `BAR` [E0507]
}
