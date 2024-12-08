// Like `projection-bound-cycle.rs` but this avoids using
// `feature(trivial_bounds)`.

trait Print {
    fn print();
}

trait Foo {
    type Item: Sized where <Self as Foo>::Item: Sized;
}

struct Number<T> { t: T }

impl<T> Foo for Number<T> {
    // Well-formedness checks require that the following
    // goal is true:
    // ```
    // if ([T]: Sized) { # if the where clauses hold
    //     [T]: Sized # then the bound on the associated type hold
    // }
    // ```
    // which it is :)
    type Item = [T] where [T]: Sized;
    //~^ ERROR overflow evaluating the requirement `<Number<T> as Foo>::Item == _`
}

struct OnlySized<T> where T: Sized { f: T }
impl<T> Print for OnlySized<T> {
    fn print() {
        println!("{}", std::mem::size_of::<T>());
    }
}

trait Bar {
    type Assoc: Print;
}

impl<T> Bar for T where T: Foo {
    // This is not ok, we need to prove `wf(<T as Foo>::Item)`, which requires
    // knowing that `<T as Foo>::Item: Sized` to satisfy the where clause. We
    // can use the bound on `Foo::Item` for this, but that requires
    // `wf(<T as Foo>::Item)`, which is an invalid cycle.
    type Assoc = OnlySized<<T as Foo>::Item>;
}

fn foo<T: Print>() {
    T::print() // oops, in fact `T = OnlySized<str>` which is ill-formed
}

fn bar<T: Bar>() {
    // we have `FromEnv(T: Bar)` hence
    // `<T as Bar>::Assoc` is well-formed and
    // `Implemented(<T as Bar>::Assoc: Print)` hold
    foo::<<T as Bar>::Assoc>()
}

fn main() {
    bar::<Number<u8>>()
}
