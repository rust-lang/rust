//! Unstable module containing the unstable contracts lang items and attribute macros.

use crate::intrinsics::const_eval_select;
pub use crate::macros::builtin::{contracts_ensures as ensures, contracts_requires as requires};

/// This function is used as part of the desugaring of the `#[ensures]` attribute.
///
/// This is an existing hack to allow users to omit the type of the return value in their ensures
/// attribute.
///
/// Ideally, rustc should be able to generate the type annotation.
/// The existing lowering logic makes it rather hard to add the explicit type annotation,
/// while the function call is fairly straight forward.
#[unstable(feature = "contracts_internals", issue = "128044" /* compiler-team#759 */)]
// Similar to `contract_check_requires`, we need to use the user-facing
// `contracts` feature rather than the perma-unstable `contracts_internals`.
// Const-checking doesn't honor allow_internal_unstable logic used by contract expansion.
#[lang = "contract_build_check_ensures"]
pub fn build_check_ensures<Ret, C, E>(cond: C) -> E
where
    C: FnOnce() -> E + Copy,
    E: Fn(&Ret) -> bool + Copy + 'static,
{
    cond()
}

/// This function is used as part of the contracts HIR lowering (desugaring) to
/// ensure contract code is only executed in a non-const environment, allowing
/// the contract to call non-const functions even when the function being
/// annotated with contracts is const itself.
///
/// The `contract` closure should execute the necessary requires check via
/// `contract_check_ensures` and return an ensures closure built by
/// `build_check_ensures`.
#[unstable(feature = "contracts_internals", issue = "128044")]
#[rustc_const_unstable(feature = "contracts", issue = "128044")]
#[lang = "contract_check_requires_and_build_ensures"]
pub const fn contract_check_requires_and_build_ensures<
    C: FnOnce() -> E + Copy,
    E: Fn(&Ret) -> bool + Copy,
    Ret,
>(
    contract: C,
) -> Option<E> {
    const_eval_select!(
        @capture[C: FnOnce() -> E + Copy, E: Fn(&Ret) -> bool + Copy, Ret] { contract: C } -> Option<E>:
        if const {
            None
        } else {
            Some(contract())
        }
    )
}
