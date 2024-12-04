//@ proc-macro: attribute-spans-preserved.rs

extern crate attribute_spans_preserved as foo;

use foo::foo;

#[ foo ( let y: u32 = "z"; ) ] //~ ERROR: mismatched types
#[ bar { let x: u32 = "y"; } ] //~ ERROR: mismatched types
fn main() {
}
