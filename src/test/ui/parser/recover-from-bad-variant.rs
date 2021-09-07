enum Enum {
    Foo { a: usize, b: usize },
    Bar(usize, usize),
}

fn main() {
    let x = Enum::Foo(a: 3, b: 4);
    //~^ ERROR invalid `struct` delimiters or `fn` call arguments
    match x {
        Enum::Foo(a, b) => {}
        //~^ ERROR expected tuple struct or tuple variant, found struct variant `Enum::Foo`
        Enum::Bar { a, b } => {}
        //~^ ERROR tuple variant `Enum::Bar` written as struct variant
    }
}
