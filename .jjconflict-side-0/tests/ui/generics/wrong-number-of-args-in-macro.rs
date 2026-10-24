// Regression test for #149559

struct Foo<T>; //~ ERROR type parameter `T` is never used

macro_rules! foo_ty {
    ($a:ty, $b:ty) => {
        Foo<a, $b>
        //~^ ERROR cannot find type `a` in this scope
        //~| ERROR struct takes 1 generic argument but 2 generic arguments were supplied
    };
}

fn foo<'a, 'b>() -> foo_ty!(&'b (), &'b ()) {}

fn main() {}
