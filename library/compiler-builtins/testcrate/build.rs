use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();

    // These platforms do not have f128 symbols available in their system libraries, so
    // skip related tests.
    if target.starts_with("arm-")
        || target.contains("apple-darwin")
        || target.contains("windows-msvc")
        // GCC and LLVM disagree on the ABI of `f16` and `f128` with MinGW. See
        // <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115054>.
        || target.contains("windows-gnu")
        // FIXME(llvm): There is an ABI incompatibility between GCC and Clang on 32-bit x86.
        // See <https://github.com/llvm/llvm-project/issues/77401>.
        || target.starts_with("i686")
        // 32-bit PowerPC gets code generated that Qemu cannot handle. See
        // <https://github.com/rust-lang/compiler-builtins/pull/606#issuecomment-2105635926>.
        || target.starts_with("powerpc-")
        // FIXME: We get different results from the builtin functions. See
        // <https://github.com/rust-lang/compiler-builtins/pull/606#issuecomment-2105657287>.
        || target.starts_with("powerpc64-")
    {
        println!("cargo:warning=using apfloat fallback for f128");
        println!("cargo:rustc-cfg=feature=\"no-sys-f128\"");
    }
}
