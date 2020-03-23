fn f(_: impl Iterator<Item = &'_ ()>) {} //~ ERROR missing lifetime specifier
fn main() {}
