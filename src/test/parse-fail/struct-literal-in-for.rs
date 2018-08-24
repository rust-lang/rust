// compile-flags: -Z parse-only

struct Foo {
    x: isize,
}

impl Foo {
    fn hi(&self) -> bool {
        true
    }
}

fn main() {
    for x in Foo {
        x: 3    //~ ERROR expected type, found `3`
    }.hi() { //~ ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `{`
        println!("yo");
    }
}
