// build-pass (FIXME(62277): could be check-pass?)
trait A<T: A<T>> {}

fn main() {}
