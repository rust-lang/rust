//@ ignore-compare-mode-next-solver (explicit)
//@ compile-flags: -Znext-solver

// Make sure that stepping into impl where-clauses of `NormalizesTo`
// goals is unproductive. This must not compile, see the inline
// comments.

trait Bound {
    fn method();
}
impl Bound for u32 {
    fn method() {}
}
trait Trait<T> {
    type Assoc: Bound;
}

struct Foo;

impl Trait<u32> for Foo {
    type Assoc = u32;
}
impl<T: Bound, U> Trait<U> for T {
    type Assoc = T;
}

fn impls_bound<T: Bound>() {
    T::method();
}

// The where-clause requires `Foo: Trait<T>` to hold to be wf.
// If stepping into where-clauses during normalization is considered
// to be productive, this would be the case:
//
// - `Foo: Trait<T>`
//   - via blanket impls, requires `Foo: Bound`
//     - via where-bound, requires `Foo eq <Foo as Trait<T>>::Assoc`
//       - normalize `<Foo as Trait<T>>::Assoc`
//         - via blanket impl, requires where-clause `Foo: Bound` -> cycle
fn generic<T>()
where
    <Foo as Trait<T>>::Assoc: Bound,
    //~^ ERROR the trait bound `Foo: Bound` is not satisfied
{
    // Requires proving `Foo: Bound` by normalizing
    // `<Foo as Trait<T>>::Assoc` to `Foo`.
    impls_bound::<Foo>();
    //~^ ERROR the trait bound `Foo: Bound` is not satisfied
}
fn main() {
    // Requires proving `<Foo as Trait<u32>>::Assoc: Bound`.
    // This is trivially true.
    generic::<u32>();
}
