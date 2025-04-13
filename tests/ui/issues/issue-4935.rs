// Regression test for issue #4935

fn foo(a: usize) {}
//~^ NOTE defined here
fn main() { foo(5, 6) }
//~^ ERROR function takes 1 argument but 2 arguments were supplied
//~| NOTE unexpected argument #2 of type `{integer}`
