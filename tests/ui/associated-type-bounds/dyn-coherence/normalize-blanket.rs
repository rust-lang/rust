//@ check-pass

// Determining that `Sub` is dyn-compatible requires normalizing `<Self as Constant>::Out`.
// This requires keep track of a `Self: MetaSized` bound, which is not yet implemented.

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
fn main() {}
