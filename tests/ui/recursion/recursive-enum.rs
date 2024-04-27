enum List<T> { Cons(T, List<T>), Nil }
//~^ ERROR recursive type `List` has infinite size

fn main() {}
