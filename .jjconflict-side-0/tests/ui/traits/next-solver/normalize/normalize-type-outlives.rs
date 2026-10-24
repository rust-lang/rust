//@ check-pass

trait Tr<'a> {
    type Assoc;
}

fn outlives<'o, T: 'o>() {}

fn foo<'a, 'b, T: Tr<'a, Assoc = ()>>() {
    outlives::<'b, T::Assoc>();
}

fn main() {}
