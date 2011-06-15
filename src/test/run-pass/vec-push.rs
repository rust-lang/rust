

fn push[T](&mutable vec[mutable? T] v, &T t) { v += [t]; }

fn main() { auto v = [1, 2, 3]; push(v, 1); }