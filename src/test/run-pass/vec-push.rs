

fn push<copy T>(&v: [const T], t: T) { v += [t]; }

fn main() { let v = [1, 2, 3]; push(v, 1); }
