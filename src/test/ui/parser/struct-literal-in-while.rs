struct Foo {
    x: isize,
}

impl Foo {
    fn hi(&self) -> bool {
        true
    }
}

fn main() {
    while Foo { //~ ERROR expected value, found struct `Foo`
        x: 3    //~ ERROR expected type, found `3`
    }.hi() { //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `{`
             //~| ERROR no method named `hi` found for type `()` in the current scope
        println!("yo");
    }
}
