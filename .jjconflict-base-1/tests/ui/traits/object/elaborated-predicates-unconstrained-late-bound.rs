// Make sure that when elaborating the principal of a dyn trait for projection predicates
//  we don't end up in a situation where we have an unconstrained late-bound lifetime in
// the output of a projection.

// Fix for <https://github.com/rust-lang/rust/issues/130347>.

trait A<T>: B<T = T> {}

trait B {
    type T;
}

struct Erase<T: ?Sized + B>(T::T);

fn main() {
    let x = {
        let x = String::from("hello");

        Erase::<dyn for<'a> A<&'a _>>(x.as_str())
        //~^ ERROR binding for associated type `T` references lifetime `'a`, which does not appear in the trait input types
    };

    dbg!(x.0);
}
