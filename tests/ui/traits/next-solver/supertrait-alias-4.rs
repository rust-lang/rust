//@ compile-flags: -Znext-solver
//@ check-pass

// Exercises the ambiguity that comes from replacing the associated types within the bounds
// that are required for a `impl Trait for dyn Trait` built-in object impl to hold.

trait Sup<T> {
    type Assoc;
}

trait Foo<A, B>: Sup<A, Assoc = A> + Sup<B, Assoc = B> {
    type Other: Bar<<Self as Sup<A>>::Assoc>;
}

trait Bar<T> {}
impl Bar<i32> for () {}

fn foo<A, B>(x: &(impl Foo<A, B> + ?Sized)) {}

fn main() {
    let x: &dyn Foo<_, _, Other = ()> = todo!();
    foo(x);
    let y: &dyn Foo<i32, u32, Other = ()> = x;
}
