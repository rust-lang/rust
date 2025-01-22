#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn rintf(x: f32) -> f32 {
    select_implementation! {
        name: rintf,
        use_arch: any(
            all(target_arch = "wasm32", intrinsics_enabled),
            all(target_arch = "aarch64", target_feature = "neon", target_endian = "little"),
        ),
        args: x,
    }

    super::generic::rint(x)
}
