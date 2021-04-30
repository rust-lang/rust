use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap();
    let cwd = env::current_dir().unwrap();

    println!("cargo:compiler-rt={}", cwd.join("compiler-rt").display());

    // Activate libm's unstable features to make full use of Nightly.
    println!("cargo:rustc-cfg=feature=\"unstable\"");

    // Emscripten's runtime includes all the builtins
    if target.contains("emscripten") {
        return;
    }

    // OpenBSD provides compiler_rt by default, use it instead of rebuilding it from source
    if target.contains("openbsd") {
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=compiler_rt");
        return;
    }

    // Forcibly enable memory intrinsics on wasm32 & SGX as we don't have a libc to
    // provide them.
    if (target.contains("wasm32") && !target.contains("wasi"))
        || (target.contains("sgx") && target.contains("fortanix"))
        || target.contains("-none")
        || target.contains("nvptx")
    {
        println!("cargo:rustc-cfg=feature=\"mem\"");
    }

    // NOTE we are going to assume that llvm-target, what determines our codegen option, matches the
    // target triple. This is usually correct for our built-in targets but can break in presence of
    // custom targets, which can have arbitrary names.
    let llvm_target = target.split('-').collect::<Vec<_>>();

    // Build missing intrinsics from compiler-rt C source code. If we're
    // mangling names though we assume that we're also in test mode so we don't
    // build anything and we rely on the upstream implementation of compiler-rt
    // functions
    if !cfg!(feature = "mangled-names") && cfg!(feature = "c") {
        // Don't use a C compiler for these targets:
        //
        // * wasm32 - clang 8 for wasm is somewhat hard to come by and it's
        //   unlikely that the C is really that much better than our own Rust.
        // * nvptx - everything is bitcode, not compatible with mixed C/Rust
        // * riscv - the rust-lang/rust distribution container doesn't have a C
        //   compiler nor is cc-rs ready for compilation to riscv (at this
        //   time). This can probably be removed in the future
        if !target.contains("wasm32") && !target.contains("nvptx") && !target.starts_with("riscv") {
            #[cfg(feature = "c")]
            c::compile(&llvm_target, &target);
        }
    }

    // To compile intrinsics.rs for thumb targets, where there is no libc
    if llvm_target[0].starts_with("thumb") {
        println!("cargo:rustc-cfg=thumb")
    }

    // compiler-rt `cfg`s away some intrinsics for thumbv6m and thumbv8m.base because
    // these targets do not have full Thumb-2 support but only original Thumb-1.
    // We have to cfg our code accordingly.
    if llvm_target[0] == "thumbv6m" || llvm_target[0] == "thumbv8m.base" {
        println!("cargo:rustc-cfg=thumb_1")
    }

    // Only emit the ARM Linux atomic emulation on pre-ARMv6 architectures.
    if llvm_target[0] == "armv4t" || llvm_target[0] == "armv5te" {
        println!("cargo:rustc-cfg=kernel_user_helpers")
    }
}

#[cfg(feature = "c")]
mod c {
    extern crate cc;

    use std::collections::BTreeMap;
    use std::env;
    use std::path::{Path, PathBuf};

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
    pub fn compile(llvm_target: &[&str], target: &String) {
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
        let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
        let mut consider_float_intrinsics = true;
        let cfg = &mut cc::Build::new();

        // AArch64 GCCs exit with an error condition when they encounter any kind of floating point
        // code if the `nofp` and/or `nosimd` compiler flags have been set.
        //
        // Therefore, evaluate if those flags are present and set a boolean that causes any
        // compiler-rt intrinsics that contain floating point source to be excluded for this target.
        if target_arch == "aarch64" {
            let cflags_key = String::from("CFLAGS_") + &(target.to_owned().replace("-", "_"));
            if let Ok(cflags_value) = env::var(cflags_key) {
                if cflags_value.contains("+nofp") || cflags_value.contains("+nosimd") {
                    consider_float_intrinsics = false;
                }
            }
        }

        cfg.warnings(false);

        if target_env == "msvc" {
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
        }

        let mut sources = Sources::new();
        sources.extend(&[
            ("__absvdi2", "absvdi2.c"),
            ("__absvsi2", "absvsi2.c"),
            ("__addvdi3", "addvdi3.c"),
            ("__addvsi3", "addvsi3.c"),
            ("apple_versioning", "apple_versioning.c"),
            ("__clzdi2", "clzdi2.c"),
            ("__clzsi2", "clzsi2.c"),
            ("__cmpdi2", "cmpdi2.c"),
            ("__ctzdi2", "ctzdi2.c"),
            ("__ctzsi2", "ctzsi2.c"),
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
                ("__divxc3", "divxc3.c"),
                ("__extendhfsf2", "extendhfsf2.c"),
                ("__muldc3", "muldc3.c"),
                ("__mulsc3", "mulsc3.c"),
                ("__mulxc3", "mulxc3.c"),
                ("__negdf2", "negdf2.c"),
                ("__negsf2", "negsf2.c"),
                ("__powixf2", "powixf2.c"),
                ("__truncdfhf2", "truncdfhf2.c"),
                ("__truncdfsf2", "truncdfsf2.c"),
                ("__truncsfhf2", "truncsfhf2.c"),
            ]);
        }

        // When compiling in rustbuild (the rust-lang/rust repo) this library
        // also needs to satisfy intrinsics that jemalloc or C in general may
        // need, so include a few more that aren't typically needed by
        // LLVM/Rust.
        if cfg!(feature = "rustbuild") {
            sources.extend(&[("__ffsdi2", "ffsdi2.c")]);
        }

        // On iOS and 32-bit OSX these are all just empty intrinsics, no need to
        // include them.
        if target_os != "ios" && (target_vendor != "apple" || target_arch != "x86") {
            sources.extend(&[
                ("__absvti2", "absvti2.c"),
                ("__addvti3", "addvti3.c"),
                ("__clzti2", "clzti2.c"),
                ("__cmpti2", "cmpti2.c"),
                ("__ctzti2", "ctzti2.c"),
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

        if target_vendor == "apple" {
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

        if target_env == "msvc" {
            if target_arch == "x86_64" {
                sources.extend(&[
                    ("__floatdisf", "x86_64/floatdisf.c"),
                    ("__floatdixf", "x86_64/floatdixf.c"),
                ]);
            }
        } else {
            // None of these seem to be used on x86_64 windows, and they've all
            // got the wrong ABI anyway, so we want to avoid them.
            if target_os != "windows" {
                if target_arch == "x86_64" {
                    sources.extend(&[
                        ("__floatdisf", "x86_64/floatdisf.c"),
                        ("__floatdixf", "x86_64/floatdixf.c"),
                        ("__floatundidf", "x86_64/floatundidf.S"),
                        ("__floatundisf", "x86_64/floatundisf.S"),
                        ("__floatundixf", "x86_64/floatundixf.S"),
                    ]);
                }
            }

            if target_arch == "x86" {
                sources.extend(&[
                    ("__ashldi3", "i386/ashldi3.S"),
                    ("__ashrdi3", "i386/ashrdi3.S"),
                    ("__divdi3", "i386/divdi3.S"),
                    ("__floatdidf", "i386/floatdidf.S"),
                    ("__floatdisf", "i386/floatdisf.S"),
                    ("__floatdixf", "i386/floatdixf.S"),
                    ("__floatundidf", "i386/floatundidf.S"),
                    ("__floatundisf", "i386/floatundisf.S"),
                    ("__floatundixf", "i386/floatundixf.S"),
                    ("__lshrdi3", "i386/lshrdi3.S"),
                    ("__moddi3", "i386/moddi3.S"),
                    ("__muldi3", "i386/muldi3.S"),
                    ("__udivdi3", "i386/udivdi3.S"),
                    ("__umoddi3", "i386/umoddi3.S"),
                ]);
            }
        }

        if target_arch == "arm" && target_os != "ios" && target_env != "msvc" {
            sources.extend(&[
                ("__aeabi_div0", "arm/aeabi_div0.c"),
                ("__aeabi_drsub", "arm/aeabi_drsub.c"),
                ("__aeabi_frsub", "arm/aeabi_frsub.c"),
                ("__bswapdi2", "arm/bswapdi2.S"),
                ("__bswapsi2", "arm/bswapsi2.S"),
                ("__clzdi2", "arm/clzdi2.S"),
                ("__clzsi2", "arm/clzsi2.S"),
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

            if target_os == "freebsd" {
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

        if target_arch == "aarch64" && consider_float_intrinsics {
            sources.extend(&[
                ("__comparetf2", "comparetf2.c"),
                ("__extenddftf2", "extenddftf2.c"),
                ("__extendsftf2", "extendsftf2.c"),
                ("__fixtfdi", "fixtfdi.c"),
                ("__fixtfsi", "fixtfsi.c"),
                ("__fixtfti", "fixtfti.c"),
                ("__fixunstfdi", "fixunstfdi.c"),
                ("__fixunstfsi", "fixunstfsi.c"),
                ("__fixunstfti", "fixunstfti.c"),
                ("__floatditf", "floatditf.c"),
                ("__floatsitf", "floatsitf.c"),
                ("__floatunditf", "floatunditf.c"),
                ("__floatunsitf", "floatunsitf.c"),
                ("__trunctfdf2", "trunctfdf2.c"),
                ("__trunctfsf2", "trunctfsf2.c"),
            ]);

            if target_os != "windows" {
                sources.extend(&[("__multc3", "multc3.c")]);
            }

            if target_env == "musl" {
                sources.extend(&[
                    ("__addtf3", "addtf3.c"),
                    ("__multf3", "multf3.c"),
                    ("__subtf3", "subtf3.c"),
                    ("__divtf3", "divtf3.c"),
                    ("__powitf2", "powitf2.c"),
                    ("__fe_getround", "fp_mode.c"),
                    ("__fe_raise_inexact", "fp_mode.c"),
                ]);
            }
        }

        if target_arch == "mips" {
            sources.extend(&[("__bswapsi2", "bswapsi2.c")]);
        }

        if target_arch == "mips64" {
            sources.extend(&[
                ("__extenddftf2", "extenddftf2.c"),
                ("__netf2", "comparetf2.c"),
                ("__addtf3", "addtf3.c"),
                ("__multf3", "multf3.c"),
                ("__subtf3", "subtf3.c"),
                ("__fixtfsi", "fixtfsi.c"),
                ("__floatsitf", "floatsitf.c"),
                ("__fixunstfsi", "fixunstfsi.c"),
                ("__floatunsitf", "floatunsitf.c"),
                ("__fe_getround", "fp_mode.c"),
                ("__divtf3", "divtf3.c"),
                ("__trunctfdf2", "trunctfdf2.c"),
            ]);
        }

        // Remove the assembly implementations that won't compile for the target
        if llvm_target[0] == "thumbv6m" || llvm_target[0] == "thumbv8m.base" {
            let mut to_remove = Vec::new();
            for (k, v) in sources.map.iter() {
                if v.ends_with(".S") {
                    to_remove.push(*k);
                }
            }
            sources.remove(&to_remove);

            // But use some generic implementations where possible
            sources.extend(&[("__clzdi2", "clzdi2.c"), ("__clzsi2", "clzsi2.c")])
        }

        if llvm_target[0] == "thumbv7m" || llvm_target[0] == "thumbv7em" {
            sources.remove(&["__aeabi_cdcmp", "__aeabi_cfcmp"]);
        }

        // When compiling the C code we require the user to tell us where the
        // source code is, and this is largely done so when we're compiling as
        // part of rust-lang/rust we can use the same llvm-project repository as
        // rust-lang/rust.
        let root = match env::var_os("RUST_COMPILER_RT_ROOT") {
            Some(s) => PathBuf::from(s),
            None => panic!("RUST_COMPILER_RT_ROOT is not set"),
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
        let src_dir = root.join("lib/builtins");
        if target_arch == "aarch64" {
            let atomics_libs = build_aarch64_out_of_line_atomics_libraries(&src_dir, cfg);
            if !atomics_libs.is_empty() {
                for library in atomics_libs {
                    cfg.object(library);
                }
                // Some run-time CPU feature detection is necessary, as well.
                sources.extend(&[("__aarch64_have_lse_atomics", "cpu_model.c")]);
            }
        }

        for (sym, src) in sources.map.iter() {
            let src = src_dir.join(src);
            cfg.file(&src);
            println!("cargo:rerun-if-changed={}", src.display());
            println!("cargo:rustc-cfg={}=\"optimized-c\"", sym);
        }

        cfg.compile("libcompiler-rt.a");
    }

    fn build_aarch64_out_of_line_atomics_libraries(
        builtins_dir: &Path,
        cfg: &cc::Build,
    ) -> Vec<PathBuf> {
        // NOTE: because we're recompiling the same source file in N different ways, building
        // serially is necessary. If we want to lift this restriction, we can either:
        // - create symlinks to lse.S and build those_(though we'd still need to pass special
        //   #define-like flags to each of these), or
        // - synthesizing tiny .S files in out/ with the proper #defines, which ultimately #include
        //   lse.S.
        // That said, it's unclear how useful this added complexity will be, so just do the simple
        // thing for now.
        let outlined_atomics_file = builtins_dir.join("aarch64/lse.S");
        println!("cargo:rerun-if-changed={}", outlined_atomics_file.display());

        let out_dir: PathBuf = env::var("OUT_DIR").unwrap().into();

        // Ideally, this would be a Vec of object files, but cc doesn't make it *entirely*
        // trivial to build an individual object.
        let mut atomics_libraries = Vec::new();
        for instruction_type in &["cas", "swp", "ldadd", "ldclr", "ldeor", "ldset"] {
            for size in &[1, 2, 4, 8, 16] {
                if *size == 16 && *instruction_type != "cas" {
                    continue;
                }

                for (model_number, model_name) in
                    &[(1, "relax"), (2, "acq"), (3, "rel"), (4, "acq_rel")]
                {
                    let library_name = format!(
                        "liboutline_atomic_helper_{}{}_{}.a",
                        instruction_type, size, model_name
                    );
                    let sym = format!("__aarch64_{}{}_{}", instruction_type, size, model_name);
                    let mut cfg = cfg.clone();

                    cfg.include(&builtins_dir)
                        .define(&format!("L_{}", instruction_type), None)
                        .define("SIZE", size.to_string().as_str())
                        .define("MODEL", model_number.to_string().as_str())
                        .file(&outlined_atomics_file);
                    cfg.compile(&library_name);

                    atomics_libraries.push(out_dir.join(library_name));
                    println!("cargo:rustc-cfg={}=\"optimized-c\"", sym);
                }
            }
        }
        atomics_libraries
    }
}
