// Need a different module so we try to build the mir for `test`
// before analyzing `mod foo`.

mod foo {
    pub trait Callable {
        fn call();
    }

    impl<V: ?Sized> Callable for () {
    //~^ ERROR the type parameter `V` is not constrained by the impl trait, self type, or predicates
        fn call() {}
    }
}
use foo::*;

fn test() -> impl Sized {
    <() as Callable>::call()
}

fn main() {}
