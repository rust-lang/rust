// Prefix in imports with empty braces should be resolved and checked privacy, stability, etc.

use foo::{}; //~ ERROR unresolved import `foo`

fn main() {}
