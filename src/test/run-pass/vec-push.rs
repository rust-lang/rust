

fn push<T: copy>(&v: [const T], t: T) { v += [t]; }

fn main() { let v = [1, 2, 3]; push(v, 1); }
