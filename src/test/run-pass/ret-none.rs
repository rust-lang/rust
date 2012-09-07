

enum option<T> { none, some(T), }

fn f<T: Copy>() -> option<T> { return none; }

fn main() { f::<int>(); }
