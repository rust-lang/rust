// error-pattern:Unsatisfied precondition

fn foo(x: int) { log(debug, x); }

fn main() { let x: int; foo(x); }
