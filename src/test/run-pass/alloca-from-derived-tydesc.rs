tag option[T] { some(T); none; }

type r[T] = {mutable v: (option[T])[]};

fn f[T]() -> [T] { ret ~[]; }

fn main() { let r: r[int] = {mutable v: ~[]}; r.v = f(); }
