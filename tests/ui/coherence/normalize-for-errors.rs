struct MyType;
trait MyTrait<S> {}

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

impl<T: Copy, S: Iterator> MyTrait<S> for (T, S::Item) {}
//~^ NOTE first implementation here
impl<S: Iterator> MyTrait<S> for (Box<<(MyType,) as Mirror>::Assoc>, S::Item) {}
//~^ ERROR conflicting implementations of trait `MyTrait<_>` for type `(Box<(MyType,)>,
//~| NOTE conflicting implementation for `(Box<(MyType,)>,
//~| NOTE upstream crates may add a new impl of trait `std::marker::Copy` for type `std::boxed::Box<(MyType,)>` in future versions
//~| NOTE upstream crates may add a new impl of trait `std::clone::Clone` for type `std::boxed::Box<(MyType,)>` in future versions

fn main() {}
