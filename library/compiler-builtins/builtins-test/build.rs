use std::collections::HashSet;

mod builtins_configure {
    include!("../compiler-builtins/configure.rs");
}

/// Features to enable
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SetCfg {
    NoSysF128,
    NoSysF128IntConvert,
    NoSysF16,
    NoSysF16F64Convert,
    NoSysF16F128Convert,
}

impl SetCfg {
    const ALL: &[Self] = &[
        Self::NoSysF128,
        Self::NoSysF128IntConvert,
        Self::NoSysF16,
        Self::NoSysF16F64Convert,
        Self::NoSysF16F128Convert,
    ];

    fn implies(self) -> &'static [Self] {
        match self {
            Self::NoSysF128 => [Self::NoSysF128IntConvert, Self::NoSysF16F128Convert].as_slice(),
            Self::NoSysF128IntConvert => [].as_slice(),
            Self::NoSysF16 => [Self::NoSysF16F64Convert, Self::NoSysF16F128Convert].as_slice(),
            Self::NoSysF16F64Convert => [].as_slice(),
            Self::NoSysF16F128Convert => [].as_slice(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::NoSysF128 => "no_sys_f128",
            Self::NoSysF128IntConvert => "no_sys_f128_int_convert",
            Self::NoSysF16F64Convert => "no_sys_f16_f64_convert",
            Self::NoSysF16F128Convert => "no_sys_f16_f128_convert",
            Self::NoSysF16 => "no_sys_f16",
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=../configure.rs");

    let target = builtins_configure::Target::from_env();
    let mut to_set = HashSet::new();

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
        to_set.insert(SetCfg::NoSysF128);
    }

    if target.arch == "x86" {
        // 32-bit x86 does not have `__fixunstfti`/`__fixtfti` but does have everything else
        to_set.insert(SetCfg::NoSysF128IntConvert);
        // FIXME: 32-bit x86 has a bug in `f128 -> f16` system libraries
        to_set.insert(SetCfg::NoSysF16F128Convert);
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
        to_set.insert(SetCfg::NoSysF16);
    }

    // These platforms are missing either `__extendhfdf2` or `__truncdfhf2`.
    if target.vendor == "apple" || target.os == "windows" {
        to_set.insert(SetCfg::NoSysF16F64Convert);
    }

    // Add implied features. Collection is required for borrows.
    to_set.extend(
        to_set
            .iter()
            .flat_map(|x| x.implies())
            .copied()
            .collect::<Vec<_>>(),
    );

    for cfg in SetCfg::ALL {
        builtins_configure::set_cfg(cfg.name(), to_set.contains(cfg));
    }

    builtins_configure::configure_aliases(&target);
}
