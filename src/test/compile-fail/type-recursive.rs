// error-pattern:illegal recursive type
type t1 = rec(int foo, t1 foolish);

fn main() {}
