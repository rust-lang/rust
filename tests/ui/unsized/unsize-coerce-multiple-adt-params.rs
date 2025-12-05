//@ check-pass

struct Foo<T, U>
where
    (T, U): Trait,
{
    f: <(T, U) as Trait>::Assoc,
}

trait Trait {
    type Assoc: ?Sized;
}

struct Count<const N: usize>;

impl<const N: usize> Trait for (i32, Count<N>) {
    type Assoc = [(); N];
}

impl<'a> Trait for (u32, ()) {
    type Assoc = [()];
}

// Test that we can unsize several trait params in creative ways.
fn unsize<const N: usize>(x: &Foo<i32, Count<N>>) -> &Foo<u32, ()> {
    x
}

fn main() {}
