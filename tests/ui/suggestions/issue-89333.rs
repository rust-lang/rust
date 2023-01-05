// check-fail
// Ensure we don't error when emitting trait bound not satisfied when self type
// has late bound var

fn main() {
    test(&|| 0); //~ ERROR the trait bound
}

trait Trait {}

fn test<T>(arg: &impl Fn() -> T) where for<'a> &'a T: Trait {}
