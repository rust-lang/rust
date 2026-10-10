// Keywords in attribute paths are printed as raw idents,
// but keywords in attribute arguments are not.

//@ pp-exact

#[rustfmt::r#final(final)]
fn main() {}
