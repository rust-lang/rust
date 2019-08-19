trait A<T>: std::ops::Add<Self> + Sized {}
trait B<T>: A<T> {}
trait C<T>: A<dyn B<T, Output=usize>> {}
//~^ ERROR the trait `B` cannot be made into an object

fn main() {}
