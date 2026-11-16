//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #140303
//@ compile-flags: -Zvalidate-mir --emit=link
//@ edition: 2018

// This only ICEd with `--emit=link` while only erroring when compiled
// with `--emit=metadata`. This is necessary to compute
// `mir_drops_elaborated_and_const_checked` even though there were
// earlier errors.
//
// This ICEd during MIR validation with the old solver when normalizing
// the return type of `b(c)`. `fn c` returns `impl Future` which normalizes
// to `{type error}`. The old solver is able to prove `impl Future: Future`,
// but `{type error}: Future` results in ambiguity.
//
// The new solver instead considers `{type error}: Future` to hold.

use std::future::Future;
struct Wrapper<T>(T);
fn a() {
    let _ = Wrapper(b(c));
}

async fn c(); // kaboom
//[next]~^ ERROR: free function without a body
fn b<T: Trait>(e: T) -> impl Sized {
    fn mk<T>() -> T { todo!() }
    mk::<<T as Trait>::Assoc>()
}
trait Trait {
    type Assoc;
}
impl<T, R> Trait for T
where
    T: Fn() -> R,
    R: Future,
{
    type Assoc = ();
}
fn main() {}
