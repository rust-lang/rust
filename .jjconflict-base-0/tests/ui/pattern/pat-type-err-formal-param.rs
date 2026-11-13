// Test the `.span_label(..)` to the type when there's a
// type error in a pattern due to a the formal parameter.

fn main() {}

struct Tuple(u8);

fn foo(Tuple(_): String) {} //~ ERROR mismatched types
