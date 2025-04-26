trait Trait {
    fn dummy(&self) { }
}

struct Foo<T:Trait> {
    x: T,
}

static X: Foo<usize> = Foo { //~ ERROR E0277
    x: 1, //~ ERROR: E0277
};

fn main() {
}
