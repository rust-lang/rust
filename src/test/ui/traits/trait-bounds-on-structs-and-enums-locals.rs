trait Trait {
    fn dummy(&self) { }
}

struct Foo<T:Trait> {
    x: T,
}

fn main() {
    let foo = Foo {
    //~^ ERROR E0277
        x: 3
    };

    let baz: Foo<usize> = loop { };
    //~^ ERROR E0277
}
