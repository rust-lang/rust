#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn rint(x: f64) -> f64 {
    select_implementation! {
        name: rint,
        use_arch: any(
            all(target_arch = "wasm32", intrinsics_enabled),
            all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"),
        ),
        args: x,
    }

    super::generic::rint(x)
}
