

tag option[T] { none; some(T); }

fn f[T]() -> option<T> { ret none; }

fn main() { f[int](); }
