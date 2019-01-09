fn bar<T: Sized>() { }
fn foo<T: ?Sized>() { bar::<T>() }
//~^ ERROR the size for values of type
fn main() { }
