// error-pattern: Unsatisfied precondition constraint
fn test(-foo: int) { assert (foo == 10); }

fn main() { let x = 10; test(x); log(debug, x); }
