//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// The new trait solver does not return region constraints if the goal
// is still ambiguous. This causes the following test to fail with ambiguity,
// even though `(): LeakCheckFailure<'!a, V>` would return `'!a: 'static`
// which would have caused a leak check failure.

trait Ambig {}
impl Ambig for u32 {}
impl Ambig for u16 {}

trait Id<T> {}
impl Id<u32> for u32 {}
impl Id<u16> for u16 {}


trait LeakCheckFailure<'a, V: ?Sized> {}
impl<V: ?Sized + Ambig> LeakCheckFailure<'static, V> for () {}

trait Trait<U, V> {}
impl<V> Trait<u32, V> for () where for<'a> (): LeakCheckFailure<'a, V> {}
impl<V> Trait<u16, V> for () {}
fn impls_trait<T: Trait<U, V>, U: Id<V>, V>() {}
fn main() {
    impls_trait::<(), _, _>()
}
