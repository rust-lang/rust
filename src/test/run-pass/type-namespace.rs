type a = {a: int};

fn a(a: a) -> int { return a.a; }

fn main() { let x: a = {a: 1}; assert (a(x) == 1); }
