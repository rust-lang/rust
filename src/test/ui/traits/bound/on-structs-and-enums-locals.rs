trait Trait {
    fn dummy(&self) { }
}

struct Foo<T:Trait> {
    x: T,
}

fn main() {
    let foo = Foo {
        x: 3
    //~^ ERROR E0277
    };

    let baz: Foo<usize> = loop { };
    //~^ ERROR E0277
}
