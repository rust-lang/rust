//@ check-pass

trait Trait: Super<Assoc: Bound> {}

trait Super {
    type Assoc;
}

trait Bound {}

fn foo<T>(x: T)
where
    T: Trait,
{
}

fn main() {}
