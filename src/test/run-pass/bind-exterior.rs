

fn foo(a: @int, b: @int) -> int { ret a + b; }

fn main() { let f1 = bind foo(@10, @12); assert (f1() == 22); }