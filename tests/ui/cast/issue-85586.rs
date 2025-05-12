// Check that errors for unresolved types in cast expressions are reported
// for the offending subexpression, not the whole cast expression.

#![allow(unused_variables)]

fn main() {
    let a = [1, 2, 3].iter().sum();
    let b = (a + 1) as usize;
    //~^ ERROR: type annotations needed [E0282]
}
