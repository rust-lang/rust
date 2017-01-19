// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Compiles the `compiler-rt` library, or at least the builtins part of it.
//!
//! Note that while compiler-rt has a build system associated with it, we
//! specifically don't use it here. The compiler-rt build system, written in
//! CMake, is actually *very* difficult to work with in terms of getting it to
//! compile on all the relevant platforms we want it to compile on. In the end
//! it became so much pain to work with local patches, work around the oddities
//! of the build system, etc, that we're just building everything by hand now.
//!
//! In general compiler-rt is just a bunch of intrinsics that are in practice
//! *very* stable. We just need to make sure that all the relevant functions and
//! such are compiled somewhere and placed in an object file somewhere.
//! Eventually, these should all be written in Rust!
//!
//! So below you'll find a listing of every single file in the compiler-rt repo
//! that we're compiling. We just reach in and compile with the `gcc` crate
//! which should have all the relevant flags and such already configured.
//!
//! The risk here is that if we update compiler-rt we may need to compile some
//! new intrinsics, but to be honest we surely don't use all of the intrinsics
//! listed below today so the likelihood of us actually needing a new intrinsic
//! is quite low. The failure case is also just that someone reports a link
//! error (if any) and then we just add it to the list. Overall, that cost is
//! far far less than working with compiler-rt's build system over time.

extern crate gcc;

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
}

fn main() {
    let target = env::var("TARGET").expect("TARGET was not set");

    // Emscripten's runtime includes all the builtins
    if target.contains("emscripten") {
        return;
    }

    let cfg = &mut gcc::Config::new();

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
        cfg.define("VISIBILITY_HIDDEN", None);
    }

    let mut sources = Sources::new();
    sources.extend(&["absvdi2.c",
                     "absvsi2.c",
                     "adddf3.c",
                     "addsf3.c",
                     "addvdi3.c",
                     "addvsi3.c",
                     "apple_versioning.c",
                     "ashldi3.c",
                     "ashrdi3.c",
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
                     "umodsi3.c"]);

    if !target.contains("ios") {
        sources.extend(&["absvti2.c",
                         "addvti3.c",
                         "ashlti3.c",
                         "ashrti3.c",
                         "clzti2.c",
                         "cmpti2.c",
                         "ctzti2.c",
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
                         "multi3.c",
                         "mulvti3.c",
                         "negti2.c",
                         "negvti2.c",
                         "parityti2.c",
                         "popcountti2.c",
                         "subvti3.c",
                         "ucmpti2.c",
                         "udivmodti4.c",
                         "udivti3.c",
                         "umodti3.c"]);
    }

    if target.contains("apple") {
        sources.extend(&["atomic_flag_clear.c",
                         "atomic_flag_clear_explicit.c",
                         "atomic_flag_test_and_set.c",
                         "atomic_flag_test_and_set_explicit.c",
                         "atomic_signal_fence.c",
                         "atomic_thread_fence.c"]);
    }

    if target.contains("msvc") {
        if target.contains("x86_64") {
            sources.extend(&["x86_64/floatdidf.c", "x86_64/floatdisf.c", "x86_64/floatdixf.c"]);
        }
    } else {
        if !target.contains("freebsd") {
            sources.extend(&["gcc_personality_v0.c"]);
        }

        if target.contains("x86_64") {
            sources.extend(&["x86_64/chkstk.S",
                             "x86_64/chkstk2.S",
                             "x86_64/floatdidf.c",
                             "x86_64/floatdisf.c",
                             "x86_64/floatdixf.c",
                             "x86_64/floatundidf.S",
                             "x86_64/floatundisf.S",
                             "x86_64/floatundixf.S"]);
        }

        if target.contains("i386") || target.contains("i586") || target.contains("i686") {
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

    if target.contains("arm") && !target.contains("ios") {
        sources.extend(&["arm/aeabi_cdcmp.S",
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
                         "arm/umodsi3.S"]);
    }

    if target.contains("armv7") {
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

    if target.contains("eabihf") {
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
                         "arm/unordsf2vfp.S"]);
    }

    if target.contains("aarch64") {
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

    for src in sources.map.values() {
        cfg.file(Path::new("../compiler-rt/lib/builtins").join(src));
    }

    cfg.compile("libcompiler-rt.a");
}
