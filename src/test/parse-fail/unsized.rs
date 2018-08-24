// compile-flags: -Z parse-only

// Test syntax checks for `type` keyword.

struct S1 for type; //~ ERROR expected `where`, `{`, `(`, or `;` after struct name, found `for`

pub fn main() {
}
