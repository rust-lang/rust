struct Foo { bar: f64, baz: i64, bat: i64 }

fn main() {
    let _ = Foo { bar: .5, baz: 42 };
    //~^ ERROR float literals must have an integer part
    //~| ERROR missing field `bat` in initializer of `Foo`
    let bar = 1.5f32;
    let _ = Foo { bar.into(), bat: -1, . };
    //~^ ERROR expected one of
    //~| ERROR missing fields `bar` and `baz` in initializer of `Foo`
    //~| ERROR expected identifier, found `.`
}
