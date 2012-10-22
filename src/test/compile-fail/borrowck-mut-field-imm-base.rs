struct Foo {
    mut x: uint
}

struct Bar {
    foo: Foo
}

fn main() {
    let mut b = Bar { foo: Foo { x: 3 } };
    let p = &b;
    let q = &mut b.foo.x;
    let r = &p.foo.x; //~ ERROR illegal borrow unless pure
    let s = &b.foo.x; //~ ERROR loan of mutable field as immutable conflicts with prior loan
    io::println(fmt!("*r = %u", *r));
    io::println(fmt!("*r = %u", *s));
    *q += 1;
    io::println(fmt!("*r = %u", *r));
    io::println(fmt!("*r = %u", *s));
}