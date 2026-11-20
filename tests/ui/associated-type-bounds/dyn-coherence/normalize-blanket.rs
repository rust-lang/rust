// Determining that `Sub` is dyn-compatible requires normalizing `<Self as Constant>::Output`.
// This requires keeping track of a `Self: MetaSized` bound, which is not yet implemented.

trait Constant {
    type Output;
}
impl<T: ?Sized> Constant for T {
    type Output = i32;
}

trait Super {
    type Assoc;
}

trait Sub: Super<Assoc = <Self as Constant>::Output> + Super<Assoc = i32> {}

fn foo(_: &dyn Sub) {}
//~^ ERROR the trait `Sub` is not dyn compatible

fn main() {}
