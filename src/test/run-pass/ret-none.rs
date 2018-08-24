// pretty-expanded FIXME #23616

enum option<T> { none, some(T), }

fn f<T>() -> option<T> { return option::none; }

pub fn main() { f::<isize>(); }
