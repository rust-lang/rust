


// xfail-stage0
type a = rec(int a);

fn a(a a) -> int { ret a.a; }

fn main() { let a x = rec(a=1); assert (a(x) == 1); }