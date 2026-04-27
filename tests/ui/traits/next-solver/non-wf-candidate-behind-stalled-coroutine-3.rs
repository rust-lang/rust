//@ check-pass
//@ compile-flags: -Znext-solver
//@ edition:2021

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/251>.
//
// This previously caused an ICE due to a non–well-formed coroutine
// hidden type failing the leak check in the next-solver.
//
// In `TypingMode::Analysis`, the problematic type is hidden behind a
// stalled coroutine candidate. However, in later passes (e.g. MIR
// validation), we eagerly normalize it. The candidate that was
// previously accepted as a solution then fails the leak check, resulting
// in broken MIR and ultimately an ICE.

trait Trait {
    type Assoc;
}
impl Trait for &'static u32 {
    type Assoc = ();
}
struct W<T: Trait>(T::Assoc);

fn prove_send_and_hide<T: Send>(x: T) -> impl Send { x }
fn as_dyn_send(_: &dyn Send) {}
pub fn main() {
    // Checking whether the cast to the trait object is correct
    // during MIR validation uses `TypingMode::PostAnalysis` and
    // therefore looks into the opaque.
    as_dyn_send(&async move {
        let opaque_ty = prove_send_and_hide(W::<&'static u32>(()));
        std::future::ready(opaque_ty).await;
    });
}
