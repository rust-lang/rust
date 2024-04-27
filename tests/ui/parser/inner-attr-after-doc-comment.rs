#![feature(lang_items)]
/**
 * My module
 */

#![recursion_limit="100"]
//~^ ERROR an inner attribute is not permitted following an outer doc comment
fn main() {}
