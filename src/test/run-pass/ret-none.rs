

enum option<T> { none; some(T); }

fn f<T: copy>() -> option<T> { ret none; }

fn main() { f::<int>(); }
