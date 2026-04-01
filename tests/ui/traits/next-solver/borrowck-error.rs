//@ compile-flags: -Znext-solver

use std::collections::HashMap;

fn foo() -> &'static HashMap<i32, i32>
{
    &HashMap::new()
    //~^ ERROR cannot return reference to temporary value
}

fn main() {}
