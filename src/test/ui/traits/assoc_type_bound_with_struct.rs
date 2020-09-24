trait Bar {
    type Baz;
}

struct Foo<T> where T: Bar, <T as Bar>::Baz: String { //~ ERROR expected trait, found struct
    t: T,
}

fn main() {}
