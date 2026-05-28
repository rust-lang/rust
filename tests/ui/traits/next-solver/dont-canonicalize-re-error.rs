//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

trait Tr<'a> {}

// Fulfillment in the new solver relies on an invariant to hold: Either
// `has_changed` is true, or computing a goal's certainty is idempotent.
// This isn't true for `ReError`, which we used to pass through in the
// canonicalizer even on input mode, which can cause a goal to go from
// ambig => pass, but we don't consider `has_changed` when the response
// only contains region constraints (since we usually uniquify regions).
//
//   In this test:
// Implicit negative coherence tries to prove `W<?0>: Constrain<'?1>`,
// which will then match with the impl below. This constrains `'?1` to
// `ReError`, but still bails w/ ambiguity bc we can't prove `?0: Sized`.
// Then, when we recompute the goal `W<?0>: Constrain<'error>`, when
// collecting ambiguities and overflows, we end up assembling a default
// error candidate w/o ambiguity, which causes the goal to pass, and ICE.
impl<'a, A> Tr<'a> for W<A> {}
struct W<A>(A);
impl<'a, A> Tr<'a> for A where A: Constrain<'a> {}
//~^ ERROR conflicting implementations of trait `Tr<'_>` for type `W<_>`

trait Constrain<'a> {}
impl<A: Sized> Constrain<'missing> for W<A> {}
//~^ ERROR use of undeclared lifetime name `'missing`

fn main() {}
