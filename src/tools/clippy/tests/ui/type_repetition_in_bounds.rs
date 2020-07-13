#[deny(clippy::type_repetition_in_bounds)]

pub fn foo<T>(_t: T)
where
    T: Copy,
    T: Clone,
{
    unimplemented!();
}

pub fn bar<T, U>(_t: T, _u: U)
where
    T: Copy,
    U: Clone,
{
    unimplemented!();
}

fn main() {}
