//! Unstable module containing the unstable contracts lang items and attribute macros.

/// Emitted by rustc as a desugaring of `#[requires(PRED)] fn foo(x: X) { ... }`
/// into: `fn foo(x: X) { check_requires(|| PRED) ... }`
#[cfg(not(bootstrap))]
#[unstable(feature = "rustc_contracts", issue = "none" /* compiler-team#759 */)]
#[lang = "contract_check_requires"]
#[track_caller]
pub fn check_requires<C: FnOnce() -> bool>(c: C) {
    if core::intrinsics::contract_checks() {
        assert!(core::intrinsics::contract_check_requires(c), "failed requires check");
    }
}

/// Emitted by rustc as a desugaring of `#[ensures(PRED)] fn foo() -> R { ... [return R;] ... }`
/// into: `fn foo() { let _check = build_check_ensures(|ret| PRED) ... [return _check(R);] ... }`
/// (including the implicit return of the tail expression, if any).
#[cfg(not(bootstrap))]
#[unstable(feature = "rustc_contracts", issue = "none" /* compiler-team#759 */)]
#[lang = "contract_build_check_ensures"]
#[track_caller]
pub fn build_check_ensures<Ret, C>(c: C) -> impl (FnOnce(Ret) -> Ret) + Copy
where
    C: for<'a> FnOnce(&'a Ret) -> bool + Copy + 'static,
{
    #[track_caller]
    move |ret| {
        if core::intrinsics::contract_checks() {
            assert!(core::intrinsics::contract_check_ensures(&ret, c), "failed ensures check");
        }
        ret
    }
}
