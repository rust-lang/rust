trait Trait {}

struct Foo<T:Trait> {
    x: T,
}

enum Bar<T:Trait> {
    ABar(isize),
    BBar(T),
    CBar(usize),
}

impl<T> Foo<T> {
//~^ ERROR `T: Trait` is not satisfied
    fn uhoh() {}
}

struct Baz {
    a: Foo<isize>, //~ ERROR E0277
}

enum Boo {
    Quux(Bar<usize>), //~ ERROR E0277
}

struct Badness<U> {
    b: Foo<U>, //~ ERROR E0277
}

enum MoreBadness<V> {
    EvenMoreBadness(Bar<V>), //~ ERROR E0277
}

struct TupleLike(
    Foo<i32>, //~ ERROR E0277
);

enum Enum {
    DictionaryLike { field: Bar<u8> }, //~ ERROR E0277
}

fn main() {
}
