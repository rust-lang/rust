// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Compilation of native dependencies like LLVM.
//!
//! Native projects like LLVM unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because thie
//! compilation takes a *very* long time but also because we don't want to
//! compile LLVM 3 times as part of a normal bootstrap (we want it cached).
//!
//! LLVM and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::path::Path;
use std::process::Command;
use std::fs::{self, File};

use build_helper::output;
use cmake;
use gcc;

use Build;
use util::{staticlib, up_to_date};

/// Compile LLVM for `target`.
pub fn llvm(build: &Build, target: &str) {
    // If we're using a custom LLVM bail out here, but we can only use a
    // custom LLVM for the build triple.
    if let Some(config) = build.config.target_config.get(target) {
        if let Some(ref s) = config.llvm_config {
            return check_llvm_version(build, s);
        }
    }

    // If the cleaning trigger is newer than our built artifacts (or if the
    // artifacts are missing) then we keep going, otherwise we bail out.
    let dst = build.llvm_out(target);
    let stamp = build.src.join("src/rustllvm/llvm-auto-clean-trigger");
    let done_stamp = dst.join("llvm-finished-building");
    build.clear_if_dirty(&dst, &stamp);
    if fs::metadata(&done_stamp).is_ok() {
        return
    }

    println!("Building LLVM for {}", target);

    let _ = fs::remove_dir_all(&dst.join("build"));
    t!(fs::create_dir_all(&dst.join("build")));
    let assertions = if build.config.llvm_assertions {"ON"} else {"OFF"};

    // http://llvm.org/docs/CMake.html
    let mut cfg = cmake::Config::new(build.src.join("src/llvm"));
    if build.config.ninja {
        cfg.generator("Ninja");
    }
    cfg.target(target)
       .host(&build.config.build)
       .out_dir(&dst)
       .profile(if build.config.llvm_optimize {"Release"} else {"Debug"})
       .define("LLVM_ENABLE_ASSERTIONS", assertions)
       .define("LLVM_TARGETS_TO_BUILD", "X86;ARM;AArch64;Mips;PowerPC")
       .define("LLVM_INCLUDE_EXAMPLES", "OFF")
       .define("LLVM_INCLUDE_TESTS", "OFF")
       .define("LLVM_INCLUDE_DOCS", "OFF")
       .define("LLVM_ENABLE_ZLIB", "OFF")
       .define("WITH_POLLY", "OFF")
       .define("LLVM_ENABLE_TERMINFO", "OFF")
       .define("LLVM_ENABLE_LIBEDIT", "OFF")
       .define("LLVM_PARALLEL_COMPILE_JOBS", build.jobs().to_string());

    if target.starts_with("i686") {
        cfg.define("LLVM_BUILD_32_BITS", "ON");
    }

    // http://llvm.org/docs/HowToCrossCompileLLVM.html
    if target != build.config.build {
        // FIXME: if the llvm root for the build triple is overridden then we
        //        should use llvm-tblgen from there, also should verify that it
        //        actually exists most of the time in normal installs of LLVM.
        let host = build.llvm_out(&build.config.build).join("bin/llvm-tblgen");
        cfg.define("CMAKE_CROSSCOMPILING", "True")
           .define("LLVM_TARGET_ARCH", target.split('-').next().unwrap())
           .define("LLVM_TABLEGEN", &host)
           .define("LLVM_DEFAULT_TARGET_TRIPLE", target);
    }

    // MSVC handles compiler business itself
    if !target.contains("msvc") {
        if build.config.ccache {
           cfg.define("CMAKE_C_COMPILER", "ccache")
              .define("CMAKE_C_COMPILER_ARG1", build.cc(target))
              .define("CMAKE_CXX_COMPILER", "ccache")
              .define("CMAKE_CXX_COMPILER_ARG1", build.cxx(target));
        } else {
           cfg.define("CMAKE_C_COMPILER", build.cc(target))
              .define("CMAKE_CXX_COMPILER", build.cxx(target));
        }
        cfg.build_arg("-j").build_arg(build.jobs().to_string());

        cfg.define("CMAKE_C_FLAGS", build.cflags(target).join(" "));
        cfg.define("CMAKE_CXX_FLAGS", build.cflags(target).join(" "));
    }

    // FIXME: we don't actually need to build all LLVM tools and all LLVM
    //        libraries here, e.g. we just want a few components and a few
    //        tools. Figure out how to filter them down and only build the right
    //        tools and libs on all platforms.
    cfg.build();

    t!(File::create(&done_stamp));
}

fn check_llvm_version(build: &Build, llvm_config: &Path) {
    if !build.config.llvm_version_check {
        return
    }

    let mut cmd = Command::new(llvm_config);
    let version = output(cmd.arg("--version"));
    if version.starts_with("3.5") || version.starts_with("3.6") ||
       version.starts_with("3.7") {
        return
    }
    panic!("\n\nbad LLVM version: {}, need >=3.5\n\n", version)
}

/// Compiles the `compiler-rt` library, or at least the builtins part of it.
///
/// Note that while compiler-rt has a build system associated with it, we
/// specifically don't use it here. The compiler-rt build system, written in
/// CMake, is actually *very* difficult to work with in terms of getting it to
/// compile on all the relevant platforms we want it to compile on. In the end
/// it became so much pain to work with local patches, work around the oddities
/// of the build system, etc, that we're just building everything by hand now.
///
/// In general compiler-rt is just a bunch of intrinsics that are in practice
/// *very* stable. We just need to make sure that all the relevant functions and
/// such are compiled somewhere and placed in an object file somewhere.
/// Eventually, these should all be written in Rust!
///
/// So below you'll find a listing of every single file in the compiler-rt repo
/// that we're compiling. We just reach in and compile with the `gcc` crate
/// which should have all the relevant flags and such already configured.
///
/// The risk here is that if we update compiler-rt we may need to compile some
/// new intrinsics, but to be honest we surely don't use all of the intrinsics
/// listed below today so the likelihood of us actually needing a new intrinsic
/// is quite low. The failure case is also just that someone reports a link
/// error (if any) and then we just add it to the list. Overall, that cost is
/// far far less than working with compiler-rt's build system over time.
pub fn compiler_rt(build: &Build, target: &str) {
    let build_dir = build.compiler_rt_out(target);
    let output = build_dir.join(staticlib("compiler-rt", target));
    build.compiler_rt_built.borrow_mut().insert(target.to_string(),
                                                output.clone());
    t!(fs::create_dir_all(&build_dir));

    let mut cfg = gcc::Config::new();
    cfg.cargo_metadata(false)
       .out_dir(&build_dir)
       .target(target)
       .host(&build.config.build)
       .opt_level(2)
       .debug(false);

    if target.contains("msvc") {
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

    let mut sources = vec![
        "absvdi2.c",
        "absvsi2.c",
        "adddf3.c",
        "addsf3.c",
        "addvdi3.c",
        "addvsi3.c",
        "apple_versioning.c",
        "ashldi3.c",
        "ashrdi3.c",
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
        "divdi3.c",
        "divmoddi4.c",
        "divmodsi4.c",
        "divsc3.c",
        "divsf3.c",
        "divsi3.c",
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
        "lshrdi3.c",
        "moddi3.c",
        "modsi3.c",
        "muldc3.c",
        "muldf3.c",
        "muldi3.c",
        "mulodi4.c",
        "mulosi4.c",
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
        "ucmpdi2.c",
        "udivdi3.c",
        "udivmoddi4.c",
        "udivmodsi4.c",
        "udivsi3.c",
        "umoddi3.c",
        "umodsi3.c",
    ];

    if !target.contains("ios") {
        sources.extend(vec![
            "absvti2.c",
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
            "umodti3.c",
        ]);
    }

    if target.contains("apple") {
        sources.extend(vec![
            "atomic_flag_clear.c",
            "atomic_flag_clear_explicit.c",
            "atomic_flag_test_and_set.c",
            "atomic_flag_test_and_set_explicit.c",
            "atomic_signal_fence.c",
            "atomic_thread_fence.c",
        ]);
    }

    if !target.contains("windows") {
        sources.push("emutls.c");
    }

    if target.contains("msvc") {
        if target.contains("x86_64") {
            sources.extend(vec![
                "x86_64/floatdidf.c",
                "x86_64/floatdisf.c",
                "x86_64/floatdixf.c",
            ]);
        }
    } else {
        sources.push("gcc_personality_v0.c");

        if target.contains("x86_64") {
            sources.extend(vec![
                "x86_64/chkstk.S",
                "x86_64/chkstk2.S",
                "x86_64/floatdidf.c",
                "x86_64/floatdisf.c",
                "x86_64/floatdixf.c",
                "x86_64/floatundidf.S",
                "x86_64/floatundisf.S",
                "x86_64/floatundixf.S",
            ]);
        }

        if target.contains("i386") ||
           target.contains("i586") ||
           target.contains("i686") {
            sources.extend(vec![
                "i386/ashldi3.S",
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
                "i386/umoddi3.S",
            ]);
        }
    }

    if target.contains("arm") && !target.contains("ios") {
        sources.extend(vec![
            "arm/aeabi_cdcmp.S",
            "arm/aeabi_cdcmpeq_check_nan.c",
            "arm/aeabi_cfcmp.S",
            "arm/aeabi_cfcmpeq_check_nan.c",
            "arm/aeabi_dcmp.S",
            "arm/aeabi_div0.c",
            "arm/aeabi_drsub.c",
            "arm/aeabi_fcmp.S",
            "arm/aeabi_frsub.c",
            "arm/aeabi_idivmod.S",
            "arm/aeabi_ldivmod.S",
            "arm/aeabi_memcmp.S",
            "arm/aeabi_memcpy.S",
            "arm/aeabi_memmove.S",
            "arm/aeabi_memset.S",
            "arm/aeabi_uidivmod.S",
            "arm/aeabi_uldivmod.S",
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
            "arm/umodsi3.S",
        ]);
    }

    if target.contains("armv7") {
        sources.extend(vec![
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
        ]);
    }

    if target.contains("eabihf") {
        sources.extend(vec![
            "arm/adddf3vfp.S",
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
            "arm/negdf2vfp.S",
            "arm/negsf2vfp.S",
            "arm/nedf2vfp.S",
            "arm/nesf2vfp.S",
            "arm/restore_vfp_d8_d15_regs.S",
            "arm/save_vfp_d8_d15_regs.S",
            "arm/subdf3vfp.S",
            "arm/subsf3vfp.S",
            "arm/truncdfsf2vfp.S",
            "arm/unorddf2vfp.S",
            "arm/unordsf2vfp.S",
        ]);
    }

    if target.contains("aarch64") {
        sources.extend(vec![
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
            "multc3.c",
            "trunctfdf2.c",
            "trunctfsf2.c",
        ]);
    }

    let mut out_of_date = false;
    for src in sources {
        let src = build.src.join("src/compiler-rt/lib/builtins").join(src);
        out_of_date = out_of_date || !up_to_date(&src, &output);
        cfg.file(src);
    }
    if !out_of_date {
        return
    }
    cfg.compile("libcompiler-rt.a");
}

/// Compiles the `rust_test_helpers.c` library which we used in various
/// `run-pass` test suites for ABI testing.
pub fn test_helpers(build: &Build, target: &str) {
    let dst = build.test_helpers_out(target);
    let src = build.src.join("src/rt/rust_test_helpers.c");
    if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
        return
    }

    println!("Building test helpers");
    t!(fs::create_dir_all(&dst));
    let mut cfg = gcc::Config::new();
    cfg.cargo_metadata(false)
       .out_dir(&dst)
       .target(target)
       .host(&build.config.build)
       .opt_level(0)
       .debug(false)
       .file(build.src.join("src/rt/rust_test_helpers.c"))
       .compile("librust_test_helpers.a");
}
