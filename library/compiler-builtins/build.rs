use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap();

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

    // Forcibly enable memory intrinsics on wasm32 as we don't have a libc to
    // provide them.
    if target.contains("wasm32") {
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
        // no C compiler for wasm
        if !target.contains("wasm32") {
            #[cfg(feature = "c")]
            c::compile(&llvm_target);
            println!("cargo:rustc-cfg=use_c");
        }
    }

    // To compile intrinsics.rs for thumb targets, where there is no libc
    if llvm_target[0].starts_with("thumb") {
        println!("cargo:rustc-cfg=thumb")
    }

    // compiler-rt `cfg`s away some intrinsics for thumbv6m because that target doesn't have full
    // THUMBv2 support. We have to cfg our code accordingly.
    if llvm_target[0] == "thumbv6m" {
        println!("cargo:rustc-cfg=thumbv6m")
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
    use std::path::Path;

    struct Sources {
        // SYMBOL -> PATH TO SOURCE
        map: BTreeMap<&'static str, &'static str>,
    }

    impl Sources {
        fn new() -> Sources {
            Sources { map: BTreeMap::new() }
        }

        fn extend(&mut self, sources: &[&'static str]) {
            // NOTE Some intrinsics have both a generic implementation (e.g.
            // `floatdidf.c`) and an arch optimized implementation
            // (`x86_64/floatdidf.c`). In those cases, we keep the arch optimized
            // implementation and discard the generic implementation. If we don't
            // and keep both implementations, the linker will yell at us about
            // duplicate symbols!
            for &src in sources {
                let symbol = Path::new(src).file_stem().unwrap().to_str().unwrap();
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
    pub fn compile(llvm_target: &[&str]) {
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
        let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();
        let target_arch_arm =
            target_arch.contains("arm") ||
            target_arch.contains("thumb");
        let cfg = &mut cc::Build::new();

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
        sources.extend(
            &[
                "absvdi2.c",
                "absvsi2.c",
                "addvdi3.c",
                "addvsi3.c",
                "apple_versioning.c",
                "clzdi2.c",
                "clzsi2.c",
                "cmpdi2.c",
                "ctzdi2.c",
                "ctzsi2.c",
                "divdc3.c",
                "divsc3.c",
                "divxc3.c",
                "extendhfsf2.c",
                "int_util.c",
                "muldc3.c",
                "mulsc3.c",
                "mulvdi3.c",
                "mulvsi3.c",
                "mulxc3.c",
                "negdf2.c",
                "negdi2.c",
                "negsf2.c",
                "negvdi2.c",
                "negvsi2.c",
                "paritydi2.c",
                "paritysi2.c",
                "popcountdi2.c",
                "popcountsi2.c",
                "powixf2.c",
                "subvdi3.c",
                "subvsi3.c",
                "truncdfhf2.c",
                "truncdfsf2.c",
                "truncsfhf2.c",
                "ucmpdi2.c",
            ],
        );

        // When compiling in rustbuild (the rust-lang/rust repo) this library
        // also needs to satisfy intrinsics that jemalloc or C in general may
        // need, so include a few more that aren't typically needed by
        // LLVM/Rust.
        if cfg!(feature = "rustbuild") {
            sources.extend(&[
                "ffsdi2.c",
            ]);
        }

        // On iOS and 32-bit OSX these are all just empty intrinsics, no need to
        // include them.
        if target_os != "ios" && (target_vendor != "apple" || target_arch != "x86") {
            sources.extend(
                &[
                    "absvti2.c",
                    "addvti3.c",
                    "clzti2.c",
                    "cmpti2.c",
                    "ctzti2.c",
                    "ffsti2.c",
                    "mulvti3.c",
                    "negti2.c",
                    "negvti2.c",
                    "parityti2.c",
                    "popcountti2.c",
                    "subvti3.c",
                    "ucmpti2.c",
                ],
            );
        }

        if target_vendor == "apple" {
            sources.extend(
                &[
                    "atomic_flag_clear.c",
                    "atomic_flag_clear_explicit.c",
                    "atomic_flag_test_and_set.c",
                    "atomic_flag_test_and_set_explicit.c",
                    "atomic_signal_fence.c",
                    "atomic_thread_fence.c",
                ],
            );
        }

        if target_env == "msvc" {
            if target_arch == "x86_64" {
                sources.extend(
                    &[
                        "x86_64/floatdisf.c",
                        "x86_64/floatdixf.c",
                    ],
                );
            }
        } else {
            // None of these seem to be used on x86_64 windows, and they've all
            // got the wrong ABI anyway, so we want to avoid them.
            if target_os != "windows" {
                if target_arch == "x86_64" {
                    sources.extend(
                        &[
                            "x86_64/floatdixf.c",
                            "x86_64/floatundidf.S",
                            "x86_64/floatundixf.S",
                        ],
                    );
                }
            }

            if target_arch == "x86" {
                sources.extend(
                    &[
                        "i386/ashldi3.S",
                        "i386/ashrdi3.S",
                        "i386/divdi3.S",
                        "i386/floatdidf.S",
                        "i386/floatdisf.S",
                        "i386/floatdixf.S",
                        "i386/floatundidf.S",
                        "i386/floatundisf.S",
                        "i386/floatundixf.S",
                        "i386/lshrdi3.S",
                        "i386/moddi3.S",
                        "i386/muldi3.S",
                        "i386/udivdi3.S",
                        "i386/umoddi3.S",
                    ],
                );
            }
        }

        if target_arch == "arm" && target_os != "ios" {
            sources.extend(
                &[
                    "arm/aeabi_div0.c",
                    "arm/aeabi_drsub.c",
                    "arm/aeabi_frsub.c",
                    "arm/bswapdi2.S",
                    "arm/bswapsi2.S",
                    "arm/clzdi2.S",
                    "arm/clzsi2.S",
                    "arm/divmodsi4.S",
                    "arm/modsi3.S",
                    "arm/switch16.S",
                    "arm/switch32.S",
                    "arm/switch8.S",
                    "arm/switchu8.S",
                    "arm/sync_synchronize.S",
                    "arm/udivmodsi4.S",
                    "arm/umodsi3.S",

                    // Exclude these two files for now even though we haven't
                    // translated their implementation into Rust yet (#173).
                    // They appear... buggy? The `udivsi3` implementation was
                    // the one that seemed buggy, but the `divsi3` file
                    // references a symbol from `udivsi3` so we compile them
                    // both with the Rust versions.
                    //
                    // Note that if these are added back they should be removed
                    // from thumbv6m below.
                    //
                    // "arm/divsi3.S",
                    // "arm/udivsi3.S",
                ],
            );

            // First of all aeabi_cdcmp and aeabi_cfcmp are never called by LLVM.
            // Second are little-endian only, so build fail on big-endian targets.
            // Temporally workaround: exclude these files for big-endian targets.
            if !llvm_target[0].starts_with("thumbeb") &&
               !llvm_target[0].starts_with("armeb") {
                sources.extend(
                    &[
                        "arm/aeabi_cdcmp.S",
                        "arm/aeabi_cdcmpeq_check_nan.c",
                        "arm/aeabi_cfcmp.S",
                        "arm/aeabi_cfcmpeq_check_nan.c",
                    ],
                );
            }
        }

        if llvm_target[0] == "armv7" {
            sources.extend(
                &[
                    "arm/sync_fetch_and_add_4.S",
                    "arm/sync_fetch_and_add_8.S",
                    "arm/sync_fetch_and_and_4.S",
                    "arm/sync_fetch_and_and_8.S",
                    "arm/sync_fetch_and_max_4.S",
                    "arm/sync_fetch_and_max_8.S",
                    "arm/sync_fetch_and_min_4.S",
                    "arm/sync_fetch_and_min_8.S",
                    "arm/sync_fetch_and_nand_4.S",
                    "arm/sync_fetch_and_nand_8.S",
                    "arm/sync_fetch_and_or_4.S",
                    "arm/sync_fetch_and_or_8.S",
                    "arm/sync_fetch_and_sub_4.S",
                    "arm/sync_fetch_and_sub_8.S",
                    "arm/sync_fetch_and_umax_4.S",
                    "arm/sync_fetch_and_umax_8.S",
                    "arm/sync_fetch_and_umin_4.S",
                    "arm/sync_fetch_and_umin_8.S",
                    "arm/sync_fetch_and_xor_4.S",
                    "arm/sync_fetch_and_xor_8.S",
                ],
            );
        }

        if llvm_target.last().unwrap().ends_with("eabihf") {
            if !llvm_target[0].starts_with("thumbv7em") {
                sources.extend(
                    &[
                        "arm/fixdfsivfp.S",
                        "arm/fixsfsivfp.S",
                        "arm/fixunsdfsivfp.S",
                        "arm/fixunssfsivfp.S",
                        "arm/floatsidfvfp.S",
                        "arm/floatsisfvfp.S",
                        "arm/floatunssidfvfp.S",
                        "arm/floatunssisfvfp.S",
                        "arm/restore_vfp_d8_d15_regs.S",
                        "arm/save_vfp_d8_d15_regs.S",
                    ],
                );
            }

            sources.extend(&["arm/negdf2vfp.S", "arm/negsf2vfp.S"]);

        }

        if target_arch == "aarch64" {
            sources.extend(
                &[
                    "comparetf2.c",
                    "extenddftf2.c",
                    "extendsftf2.c",
                    "fixtfdi.c",
                    "fixtfsi.c",
                    "fixtfti.c",
                    "fixunstfdi.c",
                    "fixunstfsi.c",
                    "fixunstfti.c",
                    "floatditf.c",
                    "floatsitf.c",
                    "floatunditf.c",
                    "floatunsitf.c",
                    "trunctfdf2.c",
                    "trunctfsf2.c",
                ],
            );

            if target_os != "windows" {
                sources.extend(&["multc3.c"]);
            }
        }

        // Remove the assembly implementations that won't compile for the target
        if llvm_target[0] == "thumbv6m" {
            sources.remove(
                &[
                    "clzdi2",
                    "clzsi2",
                    "divmodsi4",
                    "modsi3",
                    "switch16",
                    "switch32",
                    "switch8",
                    "switchu8",
                    "udivmodsi4",
                    "umodsi3",
                ],
            );

            // But use some generic implementations where possible
            sources.extend(&["clzdi2.c", "clzsi2.c"])
        }

        if llvm_target[0] == "thumbv7m" || llvm_target[0] == "thumbv7em" {
            sources.remove(&["aeabi_cdcmp", "aeabi_cfcmp"]);
        }

        // When compiling in rustbuild (the rust-lang/rust repo) this build
        // script runs from a directory other than this root directory.
        let root = if cfg!(feature = "rustbuild") {
            Path::new("../../libcompiler_builtins")
        } else {
            Path::new(".")
        };

        let src_dir = root.join("compiler-rt/lib/builtins");
        for src in sources.map.values() {
            let src = src_dir.join(src);
            cfg.file(&src);
            println!("cargo:rerun-if-changed={}", src.display());
        }

        cfg.compile("libcompiler-rt.a");
    }
}
