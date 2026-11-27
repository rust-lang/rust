//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/141478>.
// Used to ICE

fn pretty_print<'r, T: ToProviderRef<'r>>() {
    request::<T::ProviderRef, u8>()
}
fn request<T, R: RequestWithArg<PeanoSucc<T>>>() -> R::Output {
    unimplemented!()
}
trait RequestWithArg<P> {
    type Output;
}
impl<P, T> RequestWithArg<P> for T
where
    P: Tagged,
{
    type Output = ();
}
trait ToProviderRef<'r> {
    type ProviderRef: Tagged<Tag: Sized>;
}
trait Tagged {
    type Tag;
}
trait Unimplemented {}
impl<T: Unimplemented> Tagged for T {
    type Tag = ();
}
struct PeanoSucc<N>(N);
impl<P, U> Tagged for PeanoSucc<P>
where
    P: Tagged<Tag = U>,
{
    type Tag = ();
}

fn main() {}
