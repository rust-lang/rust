// Test that associated types in trait objects are not considered to be
// constrained.

#![feature(min_specialization)]

trait Specializable {
    fn f();
}

trait B<T> {
    type Y;
}

trait C {
    type Y;
}

impl<A: ?Sized> Specializable for A {
    default fn f() {}
}

impl<'a, T> Specializable for dyn B<T, Y = T> + 'a {
    //~^ ERROR specializing impl repeats parameter `T`
    fn f() {}
}

impl<'a, T> Specializable for dyn C<Y = (T, T)> + 'a {
    //~^ ERROR specializing impl repeats parameter `T`
    fn f() {}
}

fn main() {}
