// Forbidding a group (here, `unused`) overrules subsequent allowance of both
// the group, and an individual lint in the group (here, `unused_variables`);
// and, forbidding an individual lint (here, `non_snake_case`) overrules
// subsequent allowance of a lint group containing it (here, `nonstandard_style`). See
// Issue #42873.

#![forbid(unused, non_snake_case)]

#[allow(unused_variables)] //~ ERROR overruled
fn foo() {}

#[allow(unused)] //~ ERROR overruled
fn bar() {}

#[allow(nonstandard_style)] //~ ERROR overruled
fn main() {
    println!("hello forbidden world")
}
