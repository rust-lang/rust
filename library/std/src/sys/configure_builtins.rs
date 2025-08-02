/// Hook into .init_array to enable LSE atomic operations at startup, if
/// supported. outline-atomics is only enabled on aarch64-*-gnu* targets,
/// initialization is limited to those targets.
#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    target_env = "gnu",
    not(feature = "compiler-builtins-c")
))]
#[used]
#[unsafe(link_section = ".init_array.90")]
static RUST_LSE_INIT: extern "C" fn() = {
    extern "C" fn init_lse() {
        use compiler_builtins::aarch64_linux;

        use crate::arch;
        if arch::is_aarch64_feature_detected!("lse") {
            aarch64_linux::enable_lse();
        }
    }
    init_lse
};
