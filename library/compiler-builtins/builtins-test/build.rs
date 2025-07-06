use std::collections::HashSet;

mod builtins_configure {
    include!("../compiler-builtins/configure.rs");
}

/// Features to enable
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Feature {
    NoSysF128,
    NoSysF128IntConvert,
    NoSysF16,
    NoSysF16F64Convert,
    NoSysF16F128Convert,
}

impl Feature {
    fn implies(self) -> &'static [Self] {
        match self {
            Self::NoSysF128 => [Self::NoSysF128IntConvert, Self::NoSysF16F128Convert].as_slice(),
            Self::NoSysF128IntConvert => [].as_slice(),
            Self::NoSysF16 => [Self::NoSysF16F64Convert, Self::NoSysF16F128Convert].as_slice(),
            Self::NoSysF16F64Convert => [].as_slice(),
            Self::NoSysF16F128Convert => [].as_slice(),
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=../configure.rs");

    let target = builtins_configure::Target::from_env();
    let mut features = HashSet::new();

    // These platforms do not have f128 symbols available in their system libraries, so
    // skip related tests.
    if target.arch == "arm"
        || target.vendor == "apple"
        || target.env == "msvc"
        // GCC and LLVM disagree on the ABI of `f16` and `f128` with MinGW. See
        // <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115054>.
        || (target.os == "windows" && target.env == "gnu")
        // FIXME(llvm): There is an ABI incompatibility between GCC and Clang on 32-bit x86.
        // See <https://github.com/llvm/llvm-project/issues/77401>.
        || target.arch == "x86"
        // 32-bit PowerPC and 64-bit LE gets code generated that Qemu cannot handle. See
        // <https://github.com/rust-lang/compiler-builtins/pull/606#issuecomment-2105635926>.
        || target.arch == "powerpc"
        || target.arch == "powerpc64le"
        // FIXME: We get different results from the builtin functions. See
        // <https://github.com/rust-lang/compiler-builtins/pull/606#issuecomment-2105657287>.
        || target.arch == "powerpc64"
    {
        features.insert(Feature::NoSysF128);
    }

    if target.arch == "x86" {
        // 32-bit x86 does not have `__fixunstfti`/`__fixtfti` but does have everything else
        features.insert(Feature::NoSysF128IntConvert);
        // FIXME: 32-bit x86 has a bug in `f128 -> f16` system libraries
        features.insert(Feature::NoSysF16F128Convert);
    }

    // These platforms do not have f16 symbols available in their system libraries, so
    // skip related tests. Most of these are missing `f16 <-> f32` conversion routines.
    if (target.arch == "aarch64" && target.os == "linux")
        || target.arch.starts_with("arm")
        || target.arch == "powerpc"
        || target.arch == "powerpc64"
        || target.arch == "powerpc64le"
        || target.arch == "loongarch64"
        || (target.arch == "x86" && !target.has_feature("sse"))
        || target.os == "windows"
        // Linking says "error: function signature mismatch: __extendhfsf2" and seems to
        // think the signature is either `(i32) -> f32` or `(f32) -> f32`. See
        // <https://github.com/llvm/llvm-project/issues/96438>.
        || target.arch == "wasm32"
        || target.arch == "wasm64"
    {
        features.insert(Feature::NoSysF16);
    }

    // These platforms are missing either `__extendhfdf2` or `__truncdfhf2`.
    if target.vendor == "apple" || target.os == "windows" {
        features.insert(Feature::NoSysF16F64Convert);
    }

    // Add implied features. Collection is required for borrows.
    features.extend(
        features
            .iter()
            .flat_map(|x| x.implies())
            .copied()
            .collect::<Vec<_>>(),
    );

    for feature in features {
        let (name, warning) = match feature {
            Feature::NoSysF128 => ("no-sys-f128", "using apfloat fallback for f128"),
            Feature::NoSysF128IntConvert => (
                "no-sys-f128-int-convert",
                "using apfloat fallback for f128 <-> int conversions",
            ),
            Feature::NoSysF16F64Convert => (
                "no-sys-f16-f64-convert",
                "using apfloat fallback for f16 <-> f64 conversions",
            ),
            Feature::NoSysF16F128Convert => (
                "no-sys-f16-f128-convert",
                "using apfloat fallback for f16 <-> f128 conversions",
            ),
            Feature::NoSysF16 => ("no-sys-f16", "using apfloat fallback for f16"),
        };
        println!("cargo:warning={warning}");
        println!("cargo:rustc-cfg=feature=\"{name}\"");
    }

    builtins_configure::configure_aliases(&target);
}
