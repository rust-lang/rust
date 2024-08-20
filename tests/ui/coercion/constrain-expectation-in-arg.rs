//@ check-pass

trait Trait {
    type Item;
}

struct Struct<A: Trait<Item = B>, B> {
    pub field: A,
}

fn identity<T>(x: T) -> T {
    x
}

fn test<A: Trait<Item = B>, B>(x: &Struct<A, B>) {
    let x: &Struct<_, _> = identity(x);
}

fn main() {}
