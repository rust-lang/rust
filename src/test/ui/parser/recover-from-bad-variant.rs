enum Enum {
    Foo { a: usize, b: usize },
    Bar(usize, usize),
}

fn main() {
    let x = Enum::Foo(a: 3, b: 4);
    //~^ ERROR expected type, found `3`
    match x {
        Enum::Foo(a, b) => {}
        //~^ ERROR expected tuple struct/variant, found struct variant `Enum::Foo`
        Enum::Bar(a, b) => {}
    }
}
