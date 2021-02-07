// Regression test for issue #4935

fn foo(a: usize) {}
//~^ defined here
fn main() { foo(5, 6) }
//~^ ERROR arguments to this function are incorrect
