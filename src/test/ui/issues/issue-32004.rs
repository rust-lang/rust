enum Foo {
    Bar(i32),
    Baz
}

struct S;

fn main() {
    match Foo::Baz {
        Foo::Bar => {}
        //~^ ERROR expected unit struct/variant or constant, found tuple variant `Foo::Bar`
        _ => {}
    }

    match S {
        S(()) => {}
        //~^ ERROR expected tuple struct/variant, found unit struct `S`
    }
}
