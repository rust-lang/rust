struct Foo {
    mut x: uint
}

struct Bar {
    foo: Foo
}

fn main() {
    let mut b = Bar { foo: Foo { x: 3 } };
    let p = &b.foo.x;
    let q = &mut b.foo; //~ ERROR loan of mutable field as mutable conflicts with prior loan
    //~^ ERROR loan of mutable local variable as mutable conflicts with prior loan
    let r = &mut b; //~ ERROR loan of mutable local variable as mutable conflicts with prior loan
    io::println(fmt!("*p = %u", *p));
    q.x += 1;
    r.foo.x += 1;
    io::println(fmt!("*p = %u", *p));
}