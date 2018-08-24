// Prefix in imports with empty braces should be resolved and checked privacy, stability, etc.

use foo::{}; //~ ERROR cannot find module or enum `foo` in the crate root

fn main() {}
