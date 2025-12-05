/// Hook into .init_array to enable LSE atomic operations at startup, if
/// supported.
#[cfg(all(target_arch = "aarch64", target_os = "linux", not(feature = "compiler-builtins-c")))]
#[used]
#[unsafe(link_section = ".init_array.90")]
static RUST_LSE_INIT: extern "C" fn() = {
    extern "C" fn init_lse() {
        use crate::arch;

        // This is provided by compiler-builtins::aarch64_linux.
        unsafe extern "C" {
            fn __rust_enable_lse();
        }

        if arch::is_aarch64_feature_detected!("lse") {
            unsafe {
                __rust_enable_lse();
            }
        }
    }
    init_lse
};
