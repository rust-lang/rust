struct Foo { bar: f64, baz: i64, bat: i64 }

fn main() {
    let _ = Foo { bar: .5, baz: 42 };
    //~^ ERROR expected expression
    //~| ERROR missing field `bat` in initializer of `Foo`
    let bar = 1.5f32;
    let _ = Foo { bar.into(), bat: -1, . };
    //~^ ERROR expected one of
    //~| ERROR mismatched types
    //~| ERROR missing field `baz` in initializer of `Foo`
    //~| ERROR expected identifier, found `.`
}
