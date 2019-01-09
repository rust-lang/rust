// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

enum Foo {
    A(i32),
    B
}

fn match_enum() {
    let mut foo = Foo::B;
    let p = &mut foo;
    let _ = match foo {
        Foo::B => 1, //[mir]~ ERROR [E0503]
        _ => 2,
        Foo::A(x) => x //[ast]~ ERROR [E0503]
                       //[mir]~^ ERROR [E0503]
    };
    drop(p);
}


fn main() {
    let mut x = 1;
    let r = &mut x;
    let _ = match x {
        x => x + 1, //[ast]~ ERROR [E0503]
                    //[mir]~^ ERROR [E0503]
        y => y + 2, //[ast]~ ERROR [E0503]
                    //[mir]~^ ERROR [E0503]
    };
    drop(r);
}
