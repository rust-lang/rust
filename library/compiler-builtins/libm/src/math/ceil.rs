/// Ceil (f64)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn ceil(x: f64) -> f64 {
    select_implementation! {
        name: ceil,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        use_arch_required: all(target_arch = "x86", not(target_feature = "sse2")),
        args: x,
    }

    super::generic::ceil(x)
}
