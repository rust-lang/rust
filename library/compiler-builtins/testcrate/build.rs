use std::{collections::HashSet, env};

/// Features to enable
#[derive(Debug, PartialEq, Eq, Hash)]
enum Feature {
    NoSysF128,
    NoSysF128IntConvert,
    NoSysF16,
    NoSysF16F128Convert,
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let mut features = HashSet::new();

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
        features.insert(Feature::NoSysF128);
        features.insert(Feature::NoSysF128IntConvert);
        features.insert(Feature::NoSysF16F128Convert);
    }

    if target.starts_with("i586") || target.starts_with("i686") {
        // 32-bit x86 seems to not have `__fixunstfti`, but does have everything else
        features.insert(Feature::NoSysF128IntConvert);
    }

    if target.contains("-unknown-linux-") {
        // No `__extendhftf2` on x86, no `__trunctfhf2` on aarch64
        features.insert(Feature::NoSysF16F128Convert);
    }

    if target.starts_with("wasm32-") {
        // Linking says "error: function signature mismatch: __extendhfsf2" and seems to
        // think the signature is either `(i32) -> f32` or `(f32) -> f32`
        features.insert(Feature::NoSysF16);
    }

    for feature in features {
        let (name, warning) = match feature {
            Feature::NoSysF128 => ("no-sys-f128", "using apfloat fallback for f128"),
            Feature::NoSysF128IntConvert => (
                "no-sys-f128-int-convert",
                "using apfloat fallback for f128 to int conversions",
            ),
            Feature::NoSysF16F128Convert => (
                "no-sys-f16-f128-convert",
                "skipping using apfloat fallback for f16 <-> f128 conversions",
            ),
            Feature::NoSysF16 => ("no-sys-f16", "using apfloat fallback for f16"),
        };
        println!("cargo:warning={warning}");
        println!("cargo:rustc-cfg=feature=\"{name}\"");
    }
}
