

fn f[T](@T x) -> @T { ret x; }

fn main() { auto x = f(@3); log *x; }