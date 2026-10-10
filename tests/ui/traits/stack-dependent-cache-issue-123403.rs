//@ edition: 2021
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/123403>.
// Used to say the trait bound is not satisfied
struct W<T: ?Sized>(*const T);
trait Foo {}
trait Bar {}

impl<T: Bar + NotImplemented> Foo for T {}
impl Foo for W<u32> {}

impl<T: Foo + AssertInferenceConstraintsApplied> Bar for T {}
trait AssertInferenceConstraintsApplied {}
trait GuideFromEnv {}
trait ErrOnGuidance {}
impl<T: GuideFromEnv + ErrOnGuidance> AssertInferenceConstraintsApplied for T {}
impl<T> GuideFromEnv for T {}
impl ErrOnGuidance for W<u32> {}

impl<T> Bar for T
where
    W<T>: NotImplemented {}
trait NotImplemented {}

fn impls_foo<T: Foo>() {}
fn impls_bar<T: Bar>() {}

fn with_bound()
where
    W<u64>: GuideFromEnv,
{
    impls_foo::<W<_>>(); // commenting this line changes the next one to OK
    impls_bar::<W<_>>();
}

fn main() {
    with_bound();
}
