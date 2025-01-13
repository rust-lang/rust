/// Rounds the number toward 0 to the closest integral value (f64).
///
/// This effectively removes the decimal part of the number, leaving the integral part.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn trunc(x: f64) -> f64 {
    select_implementation! {
        name: trunc,
        use_arch: all(target_arch = "wasm32", intrinsics_enabled),
        args: x,
    }

    super::generic::trunc(x)
}
