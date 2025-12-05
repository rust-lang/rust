//@ check-pass

trait Trait {}

fn f<const N: usize>(_: impl Trait) {}

fn main() {}
