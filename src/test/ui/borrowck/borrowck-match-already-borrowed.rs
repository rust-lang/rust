enum Foo {
    A(i32),
    B
}

fn match_enum() {
    let mut foo = Foo::B;
    let p = &mut foo;
    let _ = match foo {
        Foo::B => 1, //~ ERROR [E0503]
        _ => 2,
        Foo::A(x) => x //~ ERROR [E0503]
    };
    drop(p);
}


fn main() {
    let mut x = 1;
    let r = &mut x;
    let _ = match x {
        x => x + 1, //~ ERROR [E0503]
        y => y + 2, //~ ERROR [E0503]
    };
    drop(r);
}
