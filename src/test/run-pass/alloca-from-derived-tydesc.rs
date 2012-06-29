enum option<T> { some(T), none, }

type r<T> = {mut v: ~[option<T>]};

fn f<T>() -> ~[T] { ret ~[]; }

fn main() { let r: r<int> = {mut v: ~[]}; r.v = f(); }
