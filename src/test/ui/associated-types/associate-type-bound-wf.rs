//Regression test for #78893

trait TwoAssocTypes {
    type X;
    type Y;
}

fn object_iterator<T: Iterator<Item = dyn Fn()>>() {}
//~^ ERROR the size for values of type `(dyn Fn() + 'static)` cannot be known at compilation time

fn parameter_iterator(_: impl Iterator<Item = impl ?Sized>) {}
//~^ ERROR the size for values of type `impl ?Sized` cannot be known at compilation time

fn unsized_object<T: TwoAssocTypes<X = (), Y = dyn Fn()>>() {}
//~^ ERROR the size for values of type `(dyn Fn() + 'static)` cannot be known at compilation time

fn main() {}
