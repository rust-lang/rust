mod configure;

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use configure::{Target, configure_aliases, configure_f16_f128};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=configure.rs");

    let target = Target::from_env();
    let cwd = env::current_dir().unwrap();

    configure_check_cfg();
    configure_f16_f128(&target);
    configure_aliases(&target);

    configure_libm(&target);

    println!("cargo:compiler-rt={}", cwd.join("compiler-rt").display());

    // Emscripten's runtime includes all the builtins
    if target.os == "emscripten" {
        return;
    }

    // OpenBSD provides compiler_rt by default, use it instead of rebuilding it from source
    if target.os == "openbsd" {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=compiler_rt");
        return;
    }

    // Forcibly enable memory intrinsics on wasm & SGX as we don't have a libc to
    // provide them.
    if (target.triple.contains("wasm") && !target.triple.contains("wasi"))
        || (target.triple.contains("sgx") && target.triple.contains("fortanix"))
        || target.triple.contains("-none")
        || target.triple.contains("nvptx")
        || target.triple.contains("uefi")
        || target.triple.contains("xous")
    {
        println!("cargo:rustc-cfg=feature=\"mem\"");
    }

    // These targets have hardware unaligned access support.
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"mem-unaligned\"))");
    if target.arch.contains("x86_64")
        || target.arch.contains("x86")
        || target.arch.contains("aarch64")
        || target.arch.contains("bpf")
    {
        println!("cargo:rustc-cfg=feature=\"mem-unaligned\"");
    }

    // NOTE we are going to assume that llvm-target, what determines our codegen option, matches the
    // target triple. This is usually correct for our built-in targets but can break in presence of
    // custom targets, which can have arbitrary names.
    let llvm_target = target.triple.split('-').collect::<Vec<_>>();

    // Build missing intrinsics from compiler-rt C source code. If we're
    // mangling names though we assume that we're also in test mode so we don't
    // build anything and we rely on the upstream implementation of compiler-rt
    // functions
    if !cfg!(feature = "mangled-names") && cfg!(feature = "c") {
        // Don't use a C compiler for these targets:
        //
        // * nvptx - everything is bitcode, not compatible with mixed C/Rust
        if !target.arch.contains("nvptx") {
            #[cfg(feature = "c")]
            c::compile(&llvm_target, &target);
        }
    }

    // Only emit the ARM Linux atomic emulation on pre-ARMv6 architectures. This
    // includes the old androideabi. It is deprecated but it is available as a
    // rustc target (arm-linux-androideabi).
    println!("cargo::rustc-check-cfg=cfg(kernel_user_helpers)");
    if llvm_target[0] == "armv4t"
        || llvm_target[0] == "armv5te"
        || target.triple == "arm-linux-androideabi"
    {
        println!("cargo:rustc-cfg=kernel_user_helpers")
    }

    if llvm_target[0].starts_with("aarch64") {
        generate_aarch64_outlined_atomics();
    }
}

/// Run configuration for `libm` since it is included directly.
///
/// Much of this is copied from `libm/configure.rs`.
fn configure_libm(target: &Target) {
    println!("cargo:rustc-check-cfg=cfg(intrinsics_enabled)");
    println!("cargo:rustc-check-cfg=cfg(arch_enabled)");
    println!("cargo:rustc-check-cfg=cfg(optimizations_enabled)");
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"unstable-public-internals\"))");

    // Always use intrinsics
    println!("cargo:rustc-cfg=intrinsics_enabled");

    // The arch module may contain assembly.
    if !cfg!(feature = "no-asm") {
        println!("cargo:rustc-cfg=arch_enabled");
    }

    println!("cargo:rustc-check-cfg=cfg(optimizations_enabled)");
    if !matches!(target.opt_level.as_str(), "0" | "1") {
        println!("cargo:rustc-cfg=optimizations_enabled");
    }

    // Config shorthands
    println!("cargo:rustc-check-cfg=cfg(x86_no_sse)");
    if target.arch == "x86" && !target.features.iter().any(|f| f == "sse") {
        // Shorthand to detect i586 targets
        println!("cargo:rustc-cfg=x86_no_sse");
    }

    println!(
        "cargo:rustc-env=CFG_CARGO_FEATURES={:?}",
        target.cargo_features
    );
    println!("cargo:rustc-env=CFG_OPT_LEVEL={}", target.opt_level);
    println!("cargo:rustc-env=CFG_TARGET_FEATURES={:?}", target.features);

    // Activate libm's unstable features to make full use of Nightly.
    println!("cargo:rustc-cfg=feature=\"unstable-intrinsics\"");
}

fn aarch64_symbol(ordering: Ordering) -> &'static str {
    match ordering {
        Ordering::Relaxed => "relax",
        Ordering::Acquire => "acq",
        Ordering::Release => "rel",
        Ordering::AcqRel => "acq_rel",
        _ => panic!("unknown symbol for {ordering:?}"),
    }
}

/// The `concat_idents` macro is extremely annoying and doesn't allow us to define new items.
/// Define them from the build script instead.
/// Note that the majority of the code is still defined in `aarch64.rs` through inline macros.
fn generate_aarch64_outlined_atomics() {
    use std::fmt::Write;
    // #[macro_export] so that we can use this in tests
    let gen_macro =
        |name| format!("#[macro_export] macro_rules! foreach_{name} {{ ($macro:path) => {{\n");

    // Generate different macros for add/clr/eor/set so that we can test them separately.
    let sym_names = ["cas", "ldadd", "ldclr", "ldeor", "ldset", "swp"];
    let mut macros = BTreeMap::new();
    for sym in sym_names {
        macros.insert(sym, gen_macro(sym));
    }

    // Only CAS supports 16 bytes, and it has a different implementation that uses a different macro.
    let mut cas16 = gen_macro("cas16");

    for ordering in [
        Ordering::Relaxed,
        Ordering::Acquire,
        Ordering::Release,
        Ordering::AcqRel,
    ] {
        let sym_ordering = aarch64_symbol(ordering);
        for size in [1, 2, 4, 8] {
            for (sym, macro_) in &mut macros {
                let name = format!("__aarch64_{sym}{size}_{sym_ordering}");
                writeln!(macro_, "$macro!( {ordering:?}, {size}, {name} );").unwrap();
            }
        }
        let name = format!("__aarch64_cas16_{sym_ordering}");
        writeln!(cas16, "$macro!( {ordering:?}, {name} );").unwrap();
    }

    let mut buf = String::new();
    for macro_def in macros.values().chain(std::iter::once(&cas16)) {
        buf += macro_def;
        buf += "}; }\n";
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::write(out_dir.join("outlined_atomics.rs"), buf).unwrap();
}

/// Emit directives for features we expect to support that aren't in `Cargo.toml`.
///
/// These are mostly cfg elements emitted by this `build.rs`.
fn configure_check_cfg() {
    // Functions where we can set the "optimized-c" flag
    const HAS_OPTIMIZED_C: &[&str] = &[
        "__ashldi3",
        "__ashlsi3",
        "__ashrdi3",
        "__ashrsi3",
        "__bswapsi2",
        "__bswapdi2",
        "__bswapti2",
        "__divdi3",
        "__divsi3",
        "__divmoddi4",
        "__divmodsi4",
        "__divmodsi4",
        "__divmodti4",
        "__lshrdi3",
        "__lshrsi3",
        "__moddi3",
        "__modsi3",
        "__muldi3",
        "__udivdi3",
        "__udivmoddi4",
        "__udivmodsi4",
        "__udivsi3",
        "__umoddi3",
        "__umodsi3",
    ];

    // Build a list of all aarch64 atomic operation functions
    let mut aarch_atomic = Vec::new();
    for aarch_op in ["cas", "ldadd", "ldclr", "ldeor", "ldset", "swp"] {
        let op_sizes = if aarch_op == "cas" {
            [1, 2, 4, 8, 16].as_slice()
        } else {
            [1, 2, 4, 8].as_slice()
        };

        for op_size in op_sizes {
            for ordering in ["relax", "acq", "rel", "acq_rel"] {
                aarch_atomic.push(format!("__aarch64_{aarch_op}{op_size}_{ordering}"));
            }
        }
    }

    for fn_name in HAS_OPTIMIZED_C
        .iter()
        .copied()
        .chain(aarch_atomic.iter().map(|s| s.as_str()))
    {
        println!("cargo::rustc-check-cfg=cfg({fn_name}, values(\"optimized-c\"))",);
    }

    // Rustc is unaware of sparc target features, but this does show up from
    // `rustc --print target-features --target sparc64-unknown-linux-gnu`.
    println!("cargo::rustc-check-cfg=cfg(target_feature, values(\"vis3\"))");

    // FIXME: these come from libm and should be changed there
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"checked\"))");
    println!("cargo::rustc-check-cfg=cfg(assert_no_panic)");
}

#[cfg(feature = "c")]
mod c {
    use std::collections::{BTreeMap, HashSet};
    use std::env;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use super::Target;

    struct Sources {
        // SYMBOL -> PATH TO SOURCE
        map: BTreeMap<&'static str, &'static str>,
    }

    impl Sources {
        fn new() -> Sources {
            Sources {
                map: BTreeMap::new(),
            }
        }

        fn extend(&mut self, sources: &[(&'static str, &'static str)]) {
            // NOTE Some intrinsics have both a generic implementation (e.g.
            // `floatdidf.c`) and an arch optimized implementation
            // (`x86_64/floatdidf.c`). In those cases, we keep the arch optimized
            // implementation and discard the generic implementation. If we don't
            // and keep both implementations, the linker will yell at us about
            // duplicate symbols!
            for (symbol, src) in sources {
                if src.contains("/") {
                    // Arch-optimized implementation (preferred)
                    self.map.insert(symbol, src);
                } else {
                    // Generic implementation
                    if !self.map.contains_key(symbol) {
                        self.map.insert(symbol, src);
                    }
                }
            }
        }

        fn remove(&mut self, symbols: &[&str]) {
            for symbol in symbols {
                self.map.remove(*symbol).unwrap();
            }
        }
    }

    /// Compile intrinsics from the compiler-rt C source code
    pub fn compile(llvm_target: &[&str], target: &Target) {
        let mut consider_float_intrinsics = true;
        let cfg = &mut cc::Build::new();

        // AArch64 GCCs exit with an error condition when they encounter any kind of floating point
        // code if the `nofp` and/or `nosimd` compiler flags have been set.
        //
        // Therefore, evaluate if those flags are present and set a boolean that causes any
        // compiler-rt intrinsics that contain floating point source to be excluded for this target.
        if target.arch == "aarch64" {
            let cflags_key = String::from("CFLAGS_") + &(target.triple.replace("-", "_"));
            if let Ok(cflags_value) = env::var(cflags_key) {
                if cflags_value.contains("+nofp") || cflags_value.contains("+nosimd") {
                    consider_float_intrinsics = false;
                }
            }
        }

        // `compiler-rt` requires `COMPILER_RT_HAS_FLOAT16` to be defined to make it use the
        // `_Float16` type for `f16` intrinsics. This shouldn't matter as all existing `f16`
        // intrinsics have been ported to Rust in `compiler-builtins` as C compilers don't
        // support `_Float16` on all targets (whereas Rust does). However, define the macro
        // anyway to prevent issues like rust#118813 and rust#123885 silently reoccuring if more
        // `f16` intrinsics get accidentally added here in the future.
        cfg.define("COMPILER_RT_HAS_FLOAT16", None);

        cfg.warnings(false);

        if target.env == "msvc" {
            // Don't pull in extra libraries on MSVC
            cfg.flag("/Zl");

            // Emulate C99 and C++11's __func__ for MSVC prior to 2013 CTP
            cfg.define("__func__", Some("__FUNCTION__"));
        } else {
            // Turn off various features of gcc and such, mostly copying
            // compiler-rt's build system already
            cfg.flag("-fno-builtin");
            cfg.flag("-fvisibility=hidden");
            cfg.flag("-ffreestanding");
            // Avoid the following warning appearing once **per file**:
            // clang: warning: optimization flag '-fomit-frame-pointer' is not supported for target 'armv7' [-Wignored-optimization-argument]
            //
            // Note that compiler-rt's build system also checks
            //
            // `check_cxx_compiler_flag(-fomit-frame-pointer COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG)`
            //
            // in https://github.com/rust-lang/compiler-rt/blob/c8fbcb3/cmake/config-ix.cmake#L19.
            cfg.flag_if_supported("-fomit-frame-pointer");
            cfg.define("VISIBILITY_HIDDEN", None);

            if let "aarch64" | "arm64ec" = target.arch.as_str() {
                // FIXME(llvm20): Older GCCs on A64 fail to build with
                // -Werror=implicit-function-declaration due to a compiler-rt bug.
                // With a newer LLVM we should be able to enable the flag everywhere.
                // https://github.com/llvm/llvm-project/commit/8aa9d6206ce55bdaaf422839c351fbd63f033b89
            } else {
                // Avoid implicitly creating references to undefined functions
                cfg.flag("-Werror=implicit-function-declaration");
            }
        }

        // int_util.c tries to include stdlib.h if `_WIN32` is defined,
        // which it is when compiling UEFI targets with clang. This is
        // at odds with compiling with `-ffreestanding`, as the header
        // may be incompatible or not present. Create a minimal stub
        // header to use instead.
        if target.os == "uefi" {
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let include_dir = out_dir.join("include");
            if !include_dir.exists() {
                fs::create_dir(&include_dir).unwrap();
            }
            fs::write(include_dir.join("stdlib.h"), "#include <stddef.h>").unwrap();
            cfg.flag(&format!("-I{}", include_dir.to_str().unwrap()));
        }

        let mut sources = Sources::new();
        sources.extend(&[
            ("__absvdi2", "absvdi2.c"),
            ("__absvsi2", "absvsi2.c"),
            ("__addvdi3", "addvdi3.c"),
            ("__addvsi3", "addvsi3.c"),
            ("__cmpdi2", "cmpdi2.c"),
            ("__int_util", "int_util.c"),
            ("__mulvdi3", "mulvdi3.c"),
            ("__mulvsi3", "mulvsi3.c"),
            ("__negdi2", "negdi2.c"),
            ("__negvdi2", "negvdi2.c"),
            ("__negvsi2", "negvsi2.c"),
            ("__paritydi2", "paritydi2.c"),
            ("__paritysi2", "paritysi2.c"),
            ("__popcountdi2", "popcountdi2.c"),
            ("__popcountsi2", "popcountsi2.c"),
            ("__subvdi3", "subvdi3.c"),
            ("__subvsi3", "subvsi3.c"),
            ("__ucmpdi2", "ucmpdi2.c"),
        ]);

        if consider_float_intrinsics {
            sources.extend(&[
                ("__divdc3", "divdc3.c"),
                ("__divsc3", "divsc3.c"),
                ("__muldc3", "muldc3.c"),
                ("__mulsc3", "mulsc3.c"),
                ("__negdf2", "negdf2.c"),
                ("__negsf2", "negsf2.c"),
            ]);
        }

        // On iOS and 32-bit OSX these are all just empty intrinsics, no need to
        // include them.
        if target.vendor != "apple" || target.arch != "x86" {
            sources.extend(&[
                ("__absvti2", "absvti2.c"),
                ("__addvti3", "addvti3.c"),
                ("__cmpti2", "cmpti2.c"),
                ("__ffsti2", "ffsti2.c"),
                ("__mulvti3", "mulvti3.c"),
                ("__negti2", "negti2.c"),
                ("__parityti2", "parityti2.c"),
                ("__popcountti2", "popcountti2.c"),
                ("__subvti3", "subvti3.c"),
                ("__ucmpti2", "ucmpti2.c"),
            ]);

            if consider_float_intrinsics {
                sources.extend(&[("__negvti2", "negvti2.c")]);
            }
        }

        if target.vendor == "apple" {
            sources.extend(&[
                ("atomic_flag_clear", "atomic_flag_clear.c"),
                ("atomic_flag_clear_explicit", "atomic_flag_clear_explicit.c"),
                ("atomic_flag_test_and_set", "atomic_flag_test_and_set.c"),
                (
                    "atomic_flag_test_and_set_explicit",
                    "atomic_flag_test_and_set_explicit.c",
                ),
                ("atomic_signal_fence", "atomic_signal_fence.c"),
                ("atomic_thread_fence", "atomic_thread_fence.c"),
            ]);
        }

        if target.env != "msvc" {
            if target.arch == "x86" {
                sources.extend(&[
                    ("__ashldi3", "i386/ashldi3.S"),
                    ("__ashrdi3", "i386/ashrdi3.S"),
                    ("__divdi3", "i386/divdi3.S"),
                    ("__lshrdi3", "i386/lshrdi3.S"),
                    ("__moddi3", "i386/moddi3.S"),
                    ("__muldi3", "i386/muldi3.S"),
                    ("__udivdi3", "i386/udivdi3.S"),
                    ("__umoddi3", "i386/umoddi3.S"),
                ]);
            }
        }

        if target.arch == "arm" && target.vendor != "apple" && target.env != "msvc" {
            sources.extend(&[
                ("__aeabi_div0", "arm/aeabi_div0.c"),
                ("__aeabi_drsub", "arm/aeabi_drsub.c"),
                ("__aeabi_frsub", "arm/aeabi_frsub.c"),
                ("__bswapdi2", "arm/bswapdi2.S"),
                ("__bswapsi2", "arm/bswapsi2.S"),
                ("__divmodsi4", "arm/divmodsi4.S"),
                ("__divsi3", "arm/divsi3.S"),
                ("__modsi3", "arm/modsi3.S"),
                ("__switch16", "arm/switch16.S"),
                ("__switch32", "arm/switch32.S"),
                ("__switch8", "arm/switch8.S"),
                ("__switchu8", "arm/switchu8.S"),
                ("__sync_synchronize", "arm/sync_synchronize.S"),
                ("__udivmodsi4", "arm/udivmodsi4.S"),
                ("__udivsi3", "arm/udivsi3.S"),
                ("__umodsi3", "arm/umodsi3.S"),
            ]);

            if target.os == "freebsd" {
                sources.extend(&[("__clear_cache", "clear_cache.c")]);
            }

            // First of all aeabi_cdcmp and aeabi_cfcmp are never called by LLVM.
            // Second are little-endian only, so build fail on big-endian targets.
            // Temporally workaround: exclude these files for big-endian targets.
            if !llvm_target[0].starts_with("thumbeb") && !llvm_target[0].starts_with("armeb") {
                sources.extend(&[
                    ("__aeabi_cdcmp", "arm/aeabi_cdcmp.S"),
                    ("__aeabi_cdcmpeq_check_nan", "arm/aeabi_cdcmpeq_check_nan.c"),
                    ("__aeabi_cfcmp", "arm/aeabi_cfcmp.S"),
                    ("__aeabi_cfcmpeq_check_nan", "arm/aeabi_cfcmpeq_check_nan.c"),
                ]);
            }
        }

        if llvm_target[0] == "armv7" {
            sources.extend(&[
                ("__sync_fetch_and_add_4", "arm/sync_fetch_and_add_4.S"),
                ("__sync_fetch_and_add_8", "arm/sync_fetch_and_add_8.S"),
                ("__sync_fetch_and_and_4", "arm/sync_fetch_and_and_4.S"),
                ("__sync_fetch_and_and_8", "arm/sync_fetch_and_and_8.S"),
                ("__sync_fetch_and_max_4", "arm/sync_fetch_and_max_4.S"),
                ("__sync_fetch_and_max_8", "arm/sync_fetch_and_max_8.S"),
                ("__sync_fetch_and_min_4", "arm/sync_fetch_and_min_4.S"),
                ("__sync_fetch_and_min_8", "arm/sync_fetch_and_min_8.S"),
                ("__sync_fetch_and_nand_4", "arm/sync_fetch_and_nand_4.S"),
                ("__sync_fetch_and_nand_8", "arm/sync_fetch_and_nand_8.S"),
                ("__sync_fetch_and_or_4", "arm/sync_fetch_and_or_4.S"),
                ("__sync_fetch_and_or_8", "arm/sync_fetch_and_or_8.S"),
                ("__sync_fetch_and_sub_4", "arm/sync_fetch_and_sub_4.S"),
                ("__sync_fetch_and_sub_8", "arm/sync_fetch_and_sub_8.S"),
                ("__sync_fetch_and_umax_4", "arm/sync_fetch_and_umax_4.S"),
                ("__sync_fetch_and_umax_8", "arm/sync_fetch_and_umax_8.S"),
                ("__sync_fetch_and_umin_4", "arm/sync_fetch_and_umin_4.S"),
                ("__sync_fetch_and_umin_8", "arm/sync_fetch_and_umin_8.S"),
                ("__sync_fetch_and_xor_4", "arm/sync_fetch_and_xor_4.S"),
                ("__sync_fetch_and_xor_8", "arm/sync_fetch_and_xor_8.S"),
            ]);
        }

        if llvm_target.last().unwrap().ends_with("eabihf") {
            if !llvm_target[0].starts_with("thumbv7em")
                && !llvm_target[0].starts_with("thumbv8m.main")
            {
                // The FPU option chosen for these architectures in cc-rs, ie:
                //     -mfpu=fpv4-sp-d16 for thumbv7em
                //     -mfpu=fpv5-sp-d16 for thumbv8m.main
                // do not support double precision floating points conversions so the files
                // that include such instructions are not included for these targets.
                sources.extend(&[
                    ("__fixdfsivfp", "arm/fixdfsivfp.S"),
                    ("__fixunsdfsivfp", "arm/fixunsdfsivfp.S"),
                    ("__floatsidfvfp", "arm/floatsidfvfp.S"),
                    ("__floatunssidfvfp", "arm/floatunssidfvfp.S"),
                ]);
            }

            sources.extend(&[
                ("__fixsfsivfp", "arm/fixsfsivfp.S"),
                ("__fixunssfsivfp", "arm/fixunssfsivfp.S"),
                ("__floatsisfvfp", "arm/floatsisfvfp.S"),
                ("__floatunssisfvfp", "arm/floatunssisfvfp.S"),
                ("__floatunssisfvfp", "arm/floatunssisfvfp.S"),
                ("__restore_vfp_d8_d15_regs", "arm/restore_vfp_d8_d15_regs.S"),
                ("__save_vfp_d8_d15_regs", "arm/save_vfp_d8_d15_regs.S"),
                ("__negdf2vfp", "arm/negdf2vfp.S"),
                ("__negsf2vfp", "arm/negsf2vfp.S"),
            ]);
        }

        if (target.arch == "aarch64" || target.arch == "arm64ec") && consider_float_intrinsics {
            sources.extend(&[
                ("__comparetf2", "comparetf2.c"),
                ("__fe_getround", "fp_mode.c"),
                ("__fe_raise_inexact", "fp_mode.c"),
            ]);

            if target.os != "windows" && target.os != "cygwin" {
                sources.extend(&[("__multc3", "multc3.c")]);
            }
        }

        if target.arch == "mips" || target.arch == "riscv32" || target.arch == "riscv64" {
            sources.extend(&[("__bswapsi2", "bswapsi2.c")]);
        }

        if target.arch == "mips64" {
            sources.extend(&[("__netf2", "comparetf2.c"), ("__fe_getround", "fp_mode.c")]);
        }

        if target.arch == "loongarch64" {
            sources.extend(&[("__netf2", "comparetf2.c"), ("__fe_getround", "fp_mode.c")]);
        }

        // Remove the assembly implementations that won't compile for the target
        if llvm_target[0] == "thumbv6m" || llvm_target[0] == "thumbv8m.base" || target.os == "uefi"
        {
            let mut to_remove = Vec::new();
            for (k, v) in sources.map.iter() {
                if v.ends_with(".S") {
                    to_remove.push(*k);
                }
            }
            sources.remove(&to_remove);
        }

        if llvm_target[0] == "thumbv7m" || llvm_target[0] == "thumbv7em" {
            sources.remove(&["__aeabi_cdcmp", "__aeabi_cfcmp"]);
        }

        // Android and Cygwin uses emulated TLS so we need a runtime support function.
        if target.os == "android" || target.os == "cygwin" {
            sources.extend(&[("__emutls_get_address", "emutls.c")]);
        }

        // Work around a bug in the NDK headers (fixed in
        // https://r.android.com/2038949 which will be released in a future
        // NDK version) by providing a definition of LONG_BIT.
        if target.os == "android" {
            cfg.define("LONG_BIT", "(8 * sizeof(long))");
        }

        // OpenHarmony also uses emulated TLS.
        if target.env == "ohos" {
            sources.extend(&[("__emutls_get_address", "emutls.c")]);
        }

        // When compiling the C code we require the user to tell us where the
        // source code is, and this is largely done so when we're compiling as
        // part of rust-lang/rust we can use the same llvm-project repository as
        // rust-lang/rust.
        let root = match env::var_os("RUST_COMPILER_RT_ROOT") {
            Some(s) => PathBuf::from(s),
            None => {
                panic!(
                    "RUST_COMPILER_RT_ROOT is not set. You may need to run \
                    `ci/download-compiler-rt.sh`."
                );
            }
        };
        if !root.exists() {
            panic!("RUST_COMPILER_RT_ROOT={} does not exist", root.display());
        }

        // Support deterministic builds by remapping the __FILE__ prefix if the
        // compiler supports it.  This fixes the nondeterminism caused by the
        // use of that macro in lib/builtins/int_util.h in compiler-rt.
        cfg.flag_if_supported(&format!("-ffile-prefix-map={}=.", root.display()));

        // Include out-of-line atomics for aarch64, which are all generated by supplying different
        // sets of flags to the same source file.
        // Note: Out-of-line aarch64 atomics are not supported by the msvc toolchain (#430) and
        // on uefi.
        let src_dir = root.join("lib/builtins");
        if target.arch == "aarch64" && target.env != "msvc" && target.os != "uefi" {
            // See below for why we're building these as separate libraries.
            build_aarch64_out_of_line_atomics_libraries(&src_dir, cfg);

            // Some run-time CPU feature detection is necessary, as well.
            let cpu_model_src = if src_dir.join("cpu_model.c").exists() {
                "cpu_model.c"
            } else {
                "cpu_model/aarch64.c"
            };
            sources.extend(&[("__aarch64_have_lse_atomics", cpu_model_src)]);
        }

        let mut added_sources = HashSet::new();
        for (sym, src) in sources.map.iter() {
            let src = src_dir.join(src);
            if added_sources.insert(src.clone()) {
                cfg.file(&src);
                println!("cargo:rerun-if-changed={}", src.display());
            }
            println!("cargo:rustc-cfg={}=\"optimized-c\"", sym);
        }

        cfg.compile("libcompiler-rt.a");
    }

    fn build_aarch64_out_of_line_atomics_libraries(builtins_dir: &Path, cfg: &mut cc::Build) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let outlined_atomics_file = builtins_dir.join("aarch64").join("lse.S");
        println!("cargo:rerun-if-changed={}", outlined_atomics_file.display());

        cfg.include(&builtins_dir);

        for instruction_type in &["cas", "swp", "ldadd", "ldclr", "ldeor", "ldset"] {
            for size in &[1, 2, 4, 8, 16] {
                if *size == 16 && *instruction_type != "cas" {
                    continue;
                }

                for (model_number, model_name) in
                    &[(1, "relax"), (2, "acq"), (3, "rel"), (4, "acq_rel")]
                {
                    // The original compiler-rt build system compiles the same
                    // source file multiple times with different compiler
                    // options. Here we do something slightly different: we
                    // create multiple .S files with the proper #defines and
                    // then include the original file.
                    //
                    // This is needed because the cc crate doesn't allow us to
                    // override the name of object files and libtool requires
                    // all objects in an archive to have unique names.
                    let path =
                        out_dir.join(format!("lse_{}{}_{}.S", instruction_type, size, model_name));
                    let mut file = File::create(&path).unwrap();
                    writeln!(file, "#define L_{}", instruction_type).unwrap();
                    writeln!(file, "#define SIZE {}", size).unwrap();
                    writeln!(file, "#define MODEL {}", model_number).unwrap();
                    writeln!(
                        file,
                        "#include \"{}\"",
                        outlined_atomics_file.canonicalize().unwrap().display()
                    )
                    .unwrap();
                    drop(file);
                    cfg.file(path);

                    let sym = format!("__aarch64_{}{}_{}", instruction_type, size, model_name);
                    println!("cargo:rustc-cfg={}=\"optimized-c\"", sym);
                }
            }
        }
    }
}
