const fn f<T>(_: Box<T>) {}
//~^ ERROR destructors cannot be evaluated at compile-time

fn main() {}
