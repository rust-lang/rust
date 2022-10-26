// check-pass
trait Foo<A> {}
impl<'a, T: Foo<A>, A> Foo<A> for &'a mut T
where
    // Needed to use `(A,)` because `A:` by itself doesn't emit a WF bound
    // as of writing this comment.
    //
    // This happens in `fn explicit_predicates_of`.
    (A,):,
{}

fn tragic<T, F: for<'a> Foo<&'a T>>(_: F) {}
fn oh_no<T, F: for<'a> Foo<&'a T>>(mut f: F) {
    // This results in a `for<'a> WF(&'a T)` bound where `'a` is replaced
    // with a placeholder before we compute the wf requirements.
    //
    // This bound would otherwise result in a `T: 'static` bound.
    tragic::<T, _>(&mut f);
}

fn main() {}
