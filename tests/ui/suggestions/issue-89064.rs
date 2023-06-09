use std::convert::TryInto;

trait A<T> {
    fn foo() {}
}

trait B<T, U> {
    fn bar() {}
}

struct S;

impl<T> A<T> for S {}
impl<T, U> B<T, U> for S {}

fn main() {
    let _ = A::foo::<S>();
    //~^ ERROR
    //~| HELP remove these generics
    //~| HELP consider moving this generic argument

    let _ = B::bar::<S, S>();
    //~^ ERROR
    //~| HELP remove these generics
    //~| HELP consider moving these generic arguments

    let _ = A::<S>::foo::<S>();
    //~^ ERROR
    //~| HELP remove these generics

    let _ = 42.into::<Option<_>>();
    //~^ ERROR
    //~| HELP remove these generics
    //~| HELP consider moving this generic argument
}
