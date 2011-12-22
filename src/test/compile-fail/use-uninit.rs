// error-pattern:Unsatisfied precondition

fn foo(x: int) { log_full(core::debug, x); }

fn main() { let x: int; foo(x); }
