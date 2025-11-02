/// Enable LSE atomic operations at startup, if supported.
///
/// Linker sections are based on what [`ctor`] does, with priorities to run slightly before user
/// code:
///
/// - Apple uses the section `__mod_init_func`, `mod_init_funcs` is needed to set
///   `S_MOD_INIT_FUNC_POINTERS`. There doesn't seem to be a way to indicate priorities.
/// - Windows uses `.CRT$XCT`, which is run before user constructors (these should use `.CRT$XCU`).
/// - ELF uses `.init_array` with a priority of 90, which runs before our `ARGV_INIT_ARRAY`
///   initializer (priority 99). Both are within the 0-100 implementation-reserved range, per docs
///   for the [`prio-ctor-dtor`] warning, and this matches compiler-rt's `CONSTRUCTOR_PRIORITY`.
///
/// To save startup time, the initializer is only run if outline atomic routines from
/// compiler-builtins may be used. If LSE is known to be available then the calls are never
/// emitted, and if we build the C intrinsics then it has its own initializer using the symbol
/// `__aarch64_have_lse_atomics`.
///
/// [`ctor`]: https://github.com/mmastrac/rust-ctor/blob/63382b833ddcbfb8b064f4e86bfa1ed4026ff356/shared/src/macros/mod.rs#L522-L534
/// [`prio-ctor-dtor`]: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "outline-atomics",
    not(target_feature = "lse"),
    not(feature = "compiler-builtins-c"),
))]
#[used]
#[cfg_attr(target_vendor = "apple", unsafe(link_section = "__DATA,__mod_init_func,mod_init_funcs"))]
#[cfg_attr(target_os = "windows", unsafe(link_section = ".CRT$XCT"))]
#[cfg_attr(
    not(any(target_vendor = "apple", target_os = "windows")),
    unsafe(link_section = ".init_array.90")
)]
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
