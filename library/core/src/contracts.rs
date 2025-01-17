//! Unstable module containing the unstable contracts lang items and attribute macros.
#![cfg(not(bootstrap))]

pub use crate::macros::builtin::{contracts_ensures as ensures, contracts_requires as requires};

/// Emitted by rustc as a desugaring of `#[ensures(PRED)] fn foo() -> R { ... [return R;] ... }`
/// into: `fn foo() { let _check = build_check_ensures(|ret| PRED) ... [return _check(R);] ... }`
/// (including the implicit return of the tail expression, if any).
#[unstable(feature = "rustc_contracts_internals", issue = "133866" /* compiler-team#759 */)]
#[lang = "contract_build_check_ensures"]
#[track_caller]
pub fn build_check_ensures<Ret, C>(cond: C) -> impl (Fn(Ret) -> Ret) + Copy
where
    C: for<'a> Fn(&'a Ret) -> bool + Copy + 'static,
{
    #[track_caller]
    move |ret| {
        crate::intrinsics::contract_check_ensures(&ret, cond);
        ret
    }
}
