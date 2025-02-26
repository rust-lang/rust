trait A<T>: std::ops::Add<Self> + Sized {}
trait B<T>: A<T> {}
trait C<T>: A<dyn B<T, Output = usize>> {}
//~^ ERROR the trait `B` is not dyn compatible
//~| ERROR the trait `B` is not dyn compatible
//~| ERROR the trait `B` is not dyn compatible

fn main() {}
