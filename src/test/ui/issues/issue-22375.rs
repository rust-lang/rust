// compile-pass
trait A<T: A<T>> {}

fn main() {}
