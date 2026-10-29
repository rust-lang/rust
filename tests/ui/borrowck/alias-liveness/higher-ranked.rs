// We once thought that we could accept this, but alas we cannot. Although in
// this case `Captures<'o>` doesn't do anything, the presence of a trait bound
// with a lifetime can result in unsoundness if the lifetime is not considered
// live. See `marker-trait-blanket-151861` and similar for examples of concrete
// unsoundness.

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

trait Outlives<'a>: 'a {}
impl<'a, T: 'a> Outlives<'a> for T {}

fn test<'o>(v: &'o Vec<i32>) -> impl Captures<'o> + for<'a> Outlives<'a> {}

fn opaque_doesnt_use_temporary() {
    let a = test(&vec![]);
    //~^ ERROR temporary value dropped while borrowed
}

fn main() {}
