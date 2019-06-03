// Test that we invoking `foo()` successfully resolves to the trait `Foo`
// (prompting the mismatched types error) but does not influence the choice
// of what kind of `Vec` we have, eventually leading to a type error.

trait Foo {
    fn foo(&self) -> isize;
}

impl Foo for Vec<usize> {
    fn foo(&self) -> isize {1}
}

impl Foo for Vec<isize> {
    fn foo(&self) -> isize {2}
}

// This is very hokey: we have heuristics to suppress messages about
// type annotations required. But placing these two bits of code into
// distinct functions, in this order, causes us to print out both
// errors I'd like to see.

fn m1() {
    // we couldn't infer the type of the vector just based on calling foo()...
    let mut x = Vec::new();
    //~^ ERROR type annotations needed
    x.foo();
}

fn m2() {
    let mut x = Vec::new();

    // ...but we still resolved `foo()` to the trait and hence know the return type.
    let y: usize = x.foo(); //~ ERROR mismatched types
}

fn main() { }
