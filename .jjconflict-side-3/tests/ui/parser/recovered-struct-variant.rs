enum Foo {
    A { a, b: usize }
    //~^ ERROR expected `:`, found `,`
}

fn main() {
    // no complaints about non-existing fields
    let f = Foo::A { a:3, b: 4};
    match f {
        // no complaints about non-existing fields
        Foo::A {a, b} => {}
    }
}
