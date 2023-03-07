#[feature(lang_items)]

#![recursion_limit="100"] //~ ERROR an inner attribute is not permitted following an outer attribute
fn main() {}
