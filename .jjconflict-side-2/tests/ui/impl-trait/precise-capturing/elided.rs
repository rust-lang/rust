//@ check-pass

fn elided(x: &()) -> impl Sized + use<'_> { x }

fn main() {}
