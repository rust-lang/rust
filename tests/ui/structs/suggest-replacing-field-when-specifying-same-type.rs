enum Foo {
    Bar { a: u8, b: i8, c: u8 },
    Baz { a: f32 },
    None,
}

fn main() {
    let foo = Foo::None;
    match foo {
        Foo::Bar { a, aa: 1, c } => (),
        //~^ ERROR variant `Foo::Bar` does not have a field named `aa` [E0026]
        //~| ERROR pattern does not mention field `b` [E0027]
        Foo::Baz { bb: 1.0 } => (),
        //~^ ERROR variant `Foo::Baz` does not have a field named `bb` [E0026]
        //~| ERROR pattern does not mention field `a` [E0027]
        _ => (),
    }

    match foo {
        Foo::Bar { a, aa: "", c } => (),
        //~^ ERROR variant `Foo::Bar` does not have a field named `aa` [E0026]
        //~| ERROR pattern does not mention field `b` [E0027]
        Foo::Baz { bb: "" } => (),
        //~^ ERROR variant `Foo::Baz` does not have a field named `bb` [E0026]
        //~| ERROR pattern does not mention field `a` [E0027]
        _ => (),
    }
}
