// error-pattern:Unsatisfied precondition

fn foo(x: int) { log x; }

fn main() { let x: int; foo(x); }