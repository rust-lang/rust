struct Foo {
    x: uint
}

struct Bar {
    foo: Foo
}

fn main() {
    let mut b = Bar { foo: Foo { x: 3 } };
    let p = &b; //~ NOTE prior loan as immutable granted here
    let q = &mut b.foo.x; //~ ERROR loan of mutable local variable as mutable conflicts with prior loan
    let r = &p.foo.x;
    io::println(fmt!("*r = %u", *r));
    *q += 1;
    io::println(fmt!("*r = %u", *r));
}