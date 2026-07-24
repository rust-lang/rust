//@ reference: expr.tuple.unary-tuple-restriction
//@ reference: type.tuple.intro
//@ reference: type.tuple.restriction
fn foo(s: &str, a: (i32, i32), s2: &str) {}

fn bar(s: &str, a: (&str,), s2: &str) {}

fn main() {
    foo("hi", 1, 2, "hi");
    //~^ ERROR function takes 3 arguments but 4 arguments were supplied
    //~| HELP: wrap these arguments in parentheses to construct a tuple
    bar("hi", "hi", "hi");
    //~^ ERROR mismatched types
    //~| HELP: use a trailing comma to create a tuple with one element
}
