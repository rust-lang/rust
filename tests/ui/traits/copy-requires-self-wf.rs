//@ check-pass

#[derive(Clone)]
struct A<'a, T>(&'a T);

impl<'a, T: Copy + 'a> Copy for A<'a, T> {}

#[derive(Clone)]
struct B<'a, T>(A<'a, T>);

// `T: '_` should be implied by `WF(B<'_, T>)`.
impl<T: Copy> Copy for B<'_, T> {}

fn main() {}
