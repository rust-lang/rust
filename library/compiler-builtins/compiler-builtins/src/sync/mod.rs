#[cfg(all(
    kernel_user_helpers,
    any(target_os = "linux", target_os = "android"),
    target_arch = "arm"
))]
pub mod arm_linux;

// Armv6k supports atomic instructions, but they are unavailable in Thumb mode
// unless Thumb-2 instructions available (v6t2).
// Using Thumb interworking allows us to use these instructions even from Thumb mode
// without Thumb-2 instructions, but LLVM does not implement that processing (as of LLVM 21),
// so we implement it here at this time.
// (`not(target_feature = "mclass")` is unneeded because v6k is not set on thumbv6m.)
#[cfg(all(
    target_arch = "arm",
    target_feature = "thumb-mode",
    target_feature = "v6k",
    not(target_feature = "v6t2"),
))]
pub mod thumbv6k;
