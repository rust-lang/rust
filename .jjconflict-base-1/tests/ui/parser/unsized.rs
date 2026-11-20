// Test syntax checks for `type` keyword.

struct S1 for type;
//~^ ERROR expected `where`, `{`, `(`, or `;` after struct name, found keyword `for`

pub fn main() {
}
