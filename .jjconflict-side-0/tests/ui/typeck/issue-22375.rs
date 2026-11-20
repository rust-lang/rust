//@ check-pass
trait A<T: A<T>> {}

fn main() {}
