//@ check-pass

pub struct Bar<'a>(&'a Self) where Self: ;

fn main() {}
