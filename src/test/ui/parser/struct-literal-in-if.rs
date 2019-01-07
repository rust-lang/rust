struct Foo {
    x: isize,
}

impl Foo {
    fn hi(&self) -> bool {
        true
    }
}

fn main() {
    if Foo { //~ ERROR expected value, found struct `Foo`
        x: 3    //~ ERROR expected type, found `3`
    }.hi() { //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `{`
        println!("yo");
    }
}
