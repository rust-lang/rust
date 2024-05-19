enum Foo {
    Bar(i32),
    Baz
}

struct S;

fn main() {
    match Foo::Baz {
        Foo::Bar => {}
        //~^ ERROR expected unit struct, unit variant or constant, found tuple variant `Foo::Bar`
        _ => {}
    }

    match S {
        S(()) => {}
        //~^ ERROR expected tuple struct or tuple variant, found unit struct `S`
    }
}
