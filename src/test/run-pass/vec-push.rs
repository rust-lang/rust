

fn push<@T>(&v: [mutable? T], t: T) { v += [t]; }

fn main() { let v = [1, 2, 3]; push(v, 1); }
