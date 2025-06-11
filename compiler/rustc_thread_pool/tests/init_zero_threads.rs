use rayon_core::ThreadPoolBuilder;

#[test]
#[cfg_attr(any(target_os = "emscripten", target_family = "wasm"), ignore)]
fn init_zero_threads() {
    ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
        .unwrap();
}
