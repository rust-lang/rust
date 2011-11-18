

tag option<T> { none; some(T); }

fn f<copy T>() -> option<T> { ret none; }

fn main() { f::<int>(); }
