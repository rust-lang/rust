//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ edition: 2021
//@ run-pass

// Regression test for <https://github.com/rust-lang/rust/issues/152735>.
//
// This used to cause a UB, due to the lack of the completeness in trait solving during
// the codegen. The trait method's where-bound can be proven in earlier passes as we
// don't look into the hidden opaque type behind the coroutine witness.
// But we do look into it during the codegen, and that fails to prove the where-bound
// because we encounter the leaking higher-ranked regions for it, which fails the
// leak-check.
//
// See <https://github.com/rust-lang/rust/issues/152735#issuecomment-4003677647> for
// more details.
//
// FIXME: We currently work around this by weakening the leak check.
// This should be revisited once the underlying implied-bounds issues
// are properly resolved.

trait Trait {
    type Assoc;
}
impl Trait for &'static u32 {
    type Assoc = ();
}
struct W<T: Trait>(T::Assoc);

fn prove_send_and_hide<T: Send>(x: T) -> impl Send { x }
trait ImpossiblePredicates<F> {
    fn call_me(&self)
    where
        F: Send,
    {
        println!("method body");
    }
}
impl<F> ImpossiblePredicates<F> for () {}

fn mk_trait_object<F>(_: F) -> Box<dyn ImpossiblePredicates<F>> {
    Box::new(())
}
pub fn main() {
    let obj = mk_trait_object(async {
        let opaque_ty = prove_send_and_hide(W::<&'static u32>(()));
        std::future::ready(opaque_ty).await;
    });
    // If we can't prove that the coroutine witness implements `Send` during codegen,
    // `impossible_predicate` would consider `call_me` as not being a valid entry of
    // the vtable.
    // That's UB.
    obj.call_me();
}
