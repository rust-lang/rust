// Regression test for issue #4935

fn foo(a: usize) {}
//~^ defined here
fn main() { foo(5, 6) }
//~^ ERROR this function takes 1 parameter but 2 parameters were supplied
