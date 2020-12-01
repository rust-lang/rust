trait Bar {
    type Baz;
}

struct Foo<T> where T: Bar, <T as Bar>::Baz: String { //~ ERROR expected trait, found struct
    t: T,
}

struct Qux<'a, T> where T: Bar, <&'a T as Bar>::Baz: String { //~ ERROR expected trait, found struct
    t: &'a T,
}

fn foo<T: Bar>(_: T) where <T as Bar>::Baz: String { //~ ERROR expected trait, found struct
}

fn qux<'a, T: Bar>(_: &'a T) where <&'a T as Bar>::Baz: String { //~ ERROR expected trait, found
}

fn main() {}
