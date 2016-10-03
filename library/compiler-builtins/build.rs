extern crate gcc;
extern crate rustc_cfg;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::{env, io, process};

use rustc_cfg::Cfg;

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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap();
    let Cfg { ref target_arch, ref target_os, ref target_env, ref target_vendor, .. } =
        Cfg::new(&target).unwrap_or_else(|e| {
            writeln!(io::stderr(), "{}", e).ok();
            process::exit(1)
        });
    // NOTE we are going to assume that llvm-target, what determines our codegen option, matches the
    // target triple. This is usually correct for our built-in targets but can break in presence of
    // custom targets, which can have arbitrary names.
    let llvm_target = target.split('-').collect::<Vec<_>>();
    let target_vendor = target_vendor.as_ref().unwrap();

    // Build missing intrinsics from compiler-rt C source code
    if env::var_os("CARGO_FEATURE_C").is_some() {
        let cfg = &mut gcc::Config::new();

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
            cfg.flag("-fomit-frame-pointer");
            cfg.flag("-ffreestanding");
        }

        // NOTE Most of the ARM intrinsics are written in assembly. Tell gcc which arch we are going to
        // target to make sure that the assembly implementations really work for the target. If the
        // implementation is not valid for the arch, then gcc will error when compiling it.
        if llvm_target[0].starts_with("thumb") {
            cfg.flag("-mthumb");

            if llvm_target.last() == Some(&"eabihf") {
                cfg.flag("-mfloat-abi=hard");
            }
        }

        if llvm_target[0] == "thumbv6m" {
            cfg.flag("-march=armv6-m");
        }

        if llvm_target[0] == "thumbv7m" {
            cfg.flag("-march=armv7-m");
        }

        if llvm_target[0] == "thumbv7em" {
            cfg.flag("-march=armv7e-m");
        }

        let mut sources = Sources::new();
        sources.extend(&["absvdi2.c",
                         "absvsi2.c",
                         "addvdi3.c",
                         "addvsi3.c",
                         "apple_versioning.c",
                         "clear_cache.c",
                         "clzdi2.c",
                         "clzsi2.c",
                         "cmpdi2.c",
                         "comparedf2.c",
                         "comparesf2.c",
                         "ctzdi2.c",
                         "ctzsi2.c",
                         "divdc3.c",
                         "divdf3.c",
                         "divsc3.c",
                         "divsf3.c",
                         "divxc3.c",
                         "extendsfdf2.c",
                         "extendhfsf2.c",
                         "ffsdi2.c",
                         "fixdfdi.c",
                         "fixdfsi.c",
                         "fixsfdi.c",
                         "fixsfsi.c",
                         "fixunsdfdi.c",
                         "fixunsdfsi.c",
                         "fixunssfdi.c",
                         "fixunssfsi.c",
                         "fixunsxfdi.c",
                         "fixunsxfsi.c",
                         "fixxfdi.c",
                         "floatdidf.c",
                         "floatdisf.c",
                         "floatdixf.c",
                         "floatsidf.c",
                         "floatsisf.c",
                         "floatundidf.c",
                         "floatundisf.c",
                         "floatundixf.c",
                         "floatunsidf.c",
                         "floatunsisf.c",
                         "int_util.c",
                         "muldc3.c",
                         "muldf3.c",
                         "muloti4.c",
                         "mulsc3.c",
                         "mulsf3.c",
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
                         "powidf2.c",
                         "powisf2.c",
                         "powixf2.c",
                         "subdf3.c",
                         "subsf3.c",
                         "subvdi3.c",
                         "subvsi3.c",
                         "truncdfhf2.c",
                         "truncdfsf2.c",
                         "truncsfhf2.c",
                         "ucmpdi2.c"]);

        if target_os != "ios" {
            sources.extend(&["absvti2.c",
                             "addtf3.c",
                             "addvti3.c",
                             "ashlti3.c",
                             "ashrti3.c",
                             "clzti2.c",
                             "cmpti2.c",
                             "ctzti2.c",
                             "divtf3.c",
                             "divti3.c",
                             "ffsti2.c",
                             "fixdfti.c",
                             "fixsfti.c",
                             "fixunsdfti.c",
                             "fixunssfti.c",
                             "fixunsxfti.c",
                             "fixxfti.c",
                             "floattidf.c",
                             "floattisf.c",
                             "floattixf.c",
                             "floatuntidf.c",
                             "floatuntisf.c",
                             "floatuntixf.c",
                             "lshrti3.c",
                             "modti3.c",
                             "multf3.c",
                             "multi3.c",
                             "mulvti3.c",
                             "negti2.c",
                             "negvti2.c",
                             "parityti2.c",
                             "popcountti2.c",
                             "powitf2.c",
                             "subtf3.c",
                             "subvti3.c",
                             "trampoline_setup.c",
                             "ucmpti2.c",
                             "udivmodti4.c",
                             "udivti3.c",
                             "umodti3.c"]);
        }

        if target_vendor == "apple" {
            sources.extend(&["atomic_flag_clear.c",
                             "atomic_flag_clear_explicit.c",
                             "atomic_flag_test_and_set.c",
                             "atomic_flag_test_and_set_explicit.c",
                             "atomic_signal_fence.c",
                             "atomic_thread_fence.c"]);
        }

        if target_os != "windows" && target_os != "none" {
            sources.extend(&["emutls.c"]);
        }

        if target_env == "msvc" {
            if target_arch == "x86_64" {
                sources.extend(&["x86_64/floatdidf.c", "x86_64/floatdisf.c", "x86_64/floatdixf.c"]);
            }
        } else {
            if target_os != "freebsd" {
                sources.extend(&["gcc_personality_v0.c"]);
            }

            if target_arch == "x86_64" {
                sources.extend(&["x86_64/chkstk.S",
                                 "x86_64/chkstk2.S",
                                 "x86_64/floatdidf.c",
                                 "x86_64/floatdisf.c",
                                 "x86_64/floatdixf.c",
                                 "x86_64/floatundidf.S",
                                 "x86_64/floatundisf.S",
                                 "x86_64/floatundixf.S"]);
            }

            if target_arch == "x86" {
                sources.extend(&["i386/ashldi3.S",
                                 "i386/ashrdi3.S",
                                 "i386/chkstk.S",
                                 "i386/chkstk2.S",
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
                                 "i386/umoddi3.S"]);
            }
        }

        if target_arch == "arm" && target_os != "ios" {
            sources.extend(&["arm/aeabi_cdcmp.S",
                             "arm/aeabi_cdcmpeq_check_nan.c",
                             "arm/aeabi_cfcmp.S",
                             "arm/aeabi_cfcmpeq_check_nan.c",
                             "arm/aeabi_dcmp.S",
                             "arm/aeabi_div0.c",
                             "arm/aeabi_drsub.c",
                             "arm/aeabi_fcmp.S",
                             "arm/aeabi_frsub.c",
                             "arm/bswapdi2.S",
                             "arm/bswapsi2.S",
                             "arm/clzdi2.S",
                             "arm/clzsi2.S",
                             "arm/comparesf2.S",
                             "arm/divmodsi4.S",
                             "arm/divsi3.S",
                             "arm/modsi3.S",
                             "arm/switch16.S",
                             "arm/switch32.S",
                             "arm/switch8.S",
                             "arm/switchu8.S",
                             "arm/sync_synchronize.S",
                             "arm/udivmodsi4.S",
                             "arm/udivsi3.S",
                             "arm/umodsi3.S"]);
        }

        if llvm_target[0] == "armv7" {
            sources.extend(&["arm/sync_fetch_and_add_4.S",
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
                             "arm/sync_fetch_and_xor_8.S"]);
        }

        if llvm_target.last().unwrap().ends_with("eabihf") {
            if !llvm_target[0].starts_with("thumbv7em") {
                sources.extend(&["arm/adddf3vfp.S",
                                 "arm/addsf3vfp.S",
                                 "arm/divdf3vfp.S",
                                 "arm/divsf3vfp.S",
                                 "arm/eqdf2vfp.S",
                                 "arm/eqsf2vfp.S",
                                 "arm/extendsfdf2vfp.S",
                                 "arm/fixdfsivfp.S",
                                 "arm/fixsfsivfp.S",
                                 "arm/fixunsdfsivfp.S",
                                 "arm/fixunssfsivfp.S",
                                 "arm/floatsidfvfp.S",
                                 "arm/floatsisfvfp.S",
                                 "arm/floatunssidfvfp.S",
                                 "arm/floatunssisfvfp.S",
                                 "arm/gedf2vfp.S",
                                 "arm/gesf2vfp.S",
                                 "arm/gtdf2vfp.S",
                                 "arm/gtsf2vfp.S",
                                 "arm/ledf2vfp.S",
                                 "arm/lesf2vfp.S",
                                 "arm/ltdf2vfp.S",
                                 "arm/ltsf2vfp.S",
                                 "arm/muldf3vfp.S",
                                 "arm/mulsf3vfp.S",
                                 "arm/nedf2vfp.S",
                                 "arm/nesf2vfp.S",
                                 "arm/restore_vfp_d8_d15_regs.S",
                                 "arm/save_vfp_d8_d15_regs.S",
                                 "arm/subdf3vfp.S",
                                 "arm/subsf3vfp.S"]);
            }

            sources.extend(&["arm/negdf2vfp.S", "arm/negsf2vfp.S"]);

        }

        if target_arch == "aarch64" {
            sources.extend(&["comparetf2.c",
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
                             "multc3.c",
                             "trunctfdf2.c",
                             "trunctfsf2.c"]);
        }

        // Remove the assembly implementations that won't compile for the target
        if llvm_target[0] == "thumbv6m" {
            sources.remove(&["aeabi_cdcmp",
                             "aeabi_cfcmp",
                             "aeabi_dcmp",
                             "aeabi_fcmp",
                             "clzdi2",
                             "clzsi2",
                             "comparesf2",
                             "divmodsi4",
                             "divsi3",
                             "modsi3",
                             "switch16",
                             "switch32",
                             "switch8",
                             "switchu8",
                             "udivmodsi4",
                             "udivsi3",
                             "umodsi3"]);

            // But use some generic implementations where possible
            sources.extend(&["clzdi2.c", "clzsi2.c"])
        }

        if llvm_target[0] == "thumbv7m" || llvm_target[0] == "thumbv7em" {
            sources.remove(&["aeabi_cdcmp", "aeabi_cfcmp"]);
        }

        for src in sources.map.values() {
            let src = Path::new("compiler-rt/compiler-rt-cdylib/compiler-rt/lib/builtins").join(src);
            cfg.file(&src);
            println!("cargo:rerun-if-changed={}", src.display());
        }

        cfg.compile("libcompiler-rt.a");
    }

    // To filter away some flaky test (see src/float/add.rs for details)
    if llvm_target.last() == Some(&"gnueabihf") {
        println!("cargo:rustc-cfg=gnueabihf")
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
}
