//@ check-pass
//@ compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/201>.
// See comment below on `fn main`.

trait Intersect<U> {
    type Output;
}

impl<T, U> Intersect<Vec<U>> for T
where
    T: Intersect<U>,
{
    type Output = T;
}

impl Intersect<Cuboid> for Cuboid {
    type Output = Cuboid;
}

fn intersect<T, U>(_: &T, _: &U) -> T::Output
where
    T: Intersect<U>,
{
    todo!()
}

struct Cuboid;
impl Cuboid {
    fn method(&self) {}
}

fn main() {
    let x = vec![];
    // Here we end up trying to normalize `<Cuboid as Intersect<Vec<?0>>>::Output`
    // for the return type of the function, to constrain `y`. The impl then requires
    // `Cuboid: Intersect<?0>`, which has two candidates. One that constrains
    // `?0 = Vec<?1>`, which itself has the same two candidates and ends up leading
    // to a recursion depth overflow. In the second impl, we constrain `?0 = Cuboid`.
    //
    // Floundering leads to us combining the overflow candidate and yes candidate to
    // overflow. Because the response was overflow, we used to bubble it up to the
    // parent normalizes-to goal, which caused us to drop its constraint that would
    // guide us to normalize the associated type mentioned before.
    //
    // After this PR, we implement a new floundering "algebra" such that `Overflow OR Maybe`
    // returns anew `Overflow { keep_constraints: true }`, which means that we don't
    // need to drop constraints in the parent normalizes-to goal. This allows us to
    // normalize `y` to `Cuboid`, and allows us to call the method successfully. We
    // then constrain the `?0` in `let x: Vec<Cuboid> = x` below, so that we don't have
    // a left over ambiguous goal.
    let y = intersect(&Cuboid, &x);
    y.method();
    let x: Vec<Cuboid> = x;
}
