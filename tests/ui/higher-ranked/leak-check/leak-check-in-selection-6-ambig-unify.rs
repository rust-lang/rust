//@ revisions: old next
//@[next] compile-flags: -Znext-solver

// The new trait solver does not return region constraints if the goal
// is still ambiguous. This should cause the following test to fail with
// ambiguity as even if  `(): LeakCheckFailure<'static, '!b, V>` unifies
// `'!b` with `'static`, we erase all region constraints.
//
// However, we do still unify the var_value for `'b` with `'static`,
// causing us to return this requirement via the `var_values` even if
// we don't return any region constraints. This is a bit inconsistent
// but isn't something we should really worry about imo.
trait Ambig {}
impl Ambig for u32 {}
impl Ambig for u16 {}

trait Id<T> {}
impl Id<u32> for u32 {}
impl Id<u16> for u16 {}


trait LeakCheckFailure<'a, 'b, V: ?Sized> {}
impl<'a, 'b: 'a, V: ?Sized + Ambig> LeakCheckFailure<'a, 'b, V> for () {}

trait Trait<U, V> {}
impl<V> Trait<u32, V> for () where for<'b> (): LeakCheckFailure<'static, 'b, V> {}
impl<V> Trait<u16, V> for () {}
fn impls_trait<T: Trait<U, V>, U: Id<V>, V>() {}
fn main() {
    impls_trait::<(), _, _>()
    //~^ ERROR type annotations needed
}
