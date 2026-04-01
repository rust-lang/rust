// Test that lifetimes must be declared for use on enums.
// See also regions-undeclared.rs

enum Yes0<'lt> {
    X3(&'lt usize)
}

enum Yes1<'a> {
    X4(&'a usize)
}

enum No0 {
    X5(&'foo usize) //~ ERROR use of undeclared lifetime name `'foo`
}

enum No1 {
    X6(&'a usize) //~ ERROR use of undeclared lifetime name `'a`
}

fn main() {}
