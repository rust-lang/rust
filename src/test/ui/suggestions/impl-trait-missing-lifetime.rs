fn f(_: impl Iterator<Item = &'_ ()>) {} //~ ERROR `'_` cannot be used here [E0637]
fn main() {}
