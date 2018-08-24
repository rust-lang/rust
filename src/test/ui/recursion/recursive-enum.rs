enum list<T> { cons(T, list<T>), nil }
//~^ ERROR recursive type `list` has infinite size

fn main() {}
