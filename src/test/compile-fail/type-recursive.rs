// xfail-stage0
// error-pattern:illegal recursive type
type t1 = {foo: int, foolish: t1};

fn main() { }