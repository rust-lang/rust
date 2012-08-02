

enum option<T> { none, some(T), }

fn f<T: copy>() -> option<T> { return none; }

fn main() { f::<int>(); }
