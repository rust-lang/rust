// Check that `where Self::Output: Copy` is turned into a bound on `Op::Output`.

//@check-pass

trait Op
where
    Self::Output: Copy,
{
    type Output;
}

fn duplicate<T: Op>(x: T::Output) -> (T::Output, T::Output) {
    (x, x)
}

fn main() {}
