//@ edition:2021
//@ compile-flags: -Copt-level=0

trait MyTrait {
    type Target: ?Sized;
}

impl<A: ?Sized> MyTrait for A {
    type Target = A;
}

fn main() {
    bug_run::<dyn MyTrait<Target = u8>>();
    //~^ ERROR the size for values of type
}

fn bug_run<T: ?Sized>()
where
    <T as MyTrait>::Target: Sized,
{
    bug::<T>();
}

fn bug<T>() {
    std::mem::size_of::<T>();
}
