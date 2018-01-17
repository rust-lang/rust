// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax_pos::symbol::Symbol;
use back::write::create_target_machine;
use llvm;
use rustc::session::Session;
use rustc::session::config::PrintRequest;
use libc::c_int;
use std::ffi::{CStr, CString};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

pub fn init(sess: &Session) {
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        static POISONED: AtomicBool = AtomicBool::new(false);
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            if llvm::LLVMStartMultithreaded() != 1 {
                // use an extra bool to make sure that all future usage of LLVM
                // cannot proceed despite the Once not running more than once.
                POISONED.store(true, Ordering::SeqCst);
            }

            configure_llvm(sess);
        });

        if POISONED.load(Ordering::SeqCst) {
            bug!("couldn't enable multi-threaded LLVM");
        }
    }
}

unsafe fn configure_llvm(sess: &Session) {
    let mut llvm_c_strs = Vec::new();
    let mut llvm_args = Vec::new();

    {
        let mut add = |arg: &str| {
            let s = CString::new(arg).unwrap();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
        };
        add("rustc"); // fake program name
        if sess.time_llvm_passes() { add("-time-passes"); }
        if sess.print_llvm_passes() { add("-debug-pass=Structure"); }

        for arg in &sess.opts.cg.llvm_args {
            add(&(*arg));
        }
    }

    llvm::LLVMInitializePasses();

    llvm::initialize_available_targets();

    llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int,
                                 llvm_args.as_ptr());
}

// WARNING: the features must be known to LLVM or the feature
// detection code will walk past the end of the feature array,
// leading to crashes.

const ARM_WHITELIST: &'static [&'static str] = &["neon\0", "vfp2\0", "vfp3\0", "vfp4\0"];

const AARCH64_WHITELIST: &'static [&'static str] = &["neon\0"];

const X86_WHITELIST: &'static [&'static str] = &["avx\0", "avx2\0", "bmi\0", "bmi2\0", "sse\0",
                                                 "sse2\0", "sse3\0", "sse4.1\0", "sse4.2\0",
                                                 "ssse3\0", "tbm\0", "lzcnt\0", "popcnt\0",
                                                 "sse4a\0", "rdrnd\0", "rdseed\0", "fma\0",
                                                 "xsave\0", "xsaveopt\0", "xsavec\0",
                                                 "xsaves\0",
                                                 "avx512bw\0", "avx512cd\0",
                                                 "avx512dq\0", "avx512er\0",
                                                 "avx512f\0", "avx512ifma\0",
                                                 "avx512pf\0", "avx512vbmi\0",
                                                 "avx512vl\0", "avx512vpopcntdq\0",
                                                 "mmx\0", "fxsr\0"];

const HEXAGON_WHITELIST: &'static [&'static str] = &["hvx\0", "hvx-double\0"];

const POWERPC_WHITELIST: &'static [&'static str] = &["altivec\0",
                                                     "power8-altivec\0", "power9-altivec\0",
                                                     "power8-vector\0", "power9-vector\0",
                                                     "vsx\0"];

const MIPS_WHITELIST: &'static [&'static str] = &["msa\0"];

pub fn target_features(sess: &Session) -> Vec<Symbol> {
    let whitelist = target_feature_whitelist(sess);
    let target_machine = create_target_machine(sess);
    let mut features = Vec::new();
    for feat in whitelist {
        if unsafe { llvm::LLVMRustHasFeature(target_machine, feat.as_ptr()) } {
            features.push(Symbol::intern(feat.to_str().unwrap()));
        }
    }
    features
}

pub fn target_feature_whitelist(sess: &Session) -> Vec<&CStr> {
    let whitelist = match &*sess.target.target.arch {
        "arm" => ARM_WHITELIST,
        "aarch64" => AARCH64_WHITELIST,
        "x86" | "x86_64" => X86_WHITELIST,
        "hexagon" => HEXAGON_WHITELIST,
        "mips" | "mips64" => MIPS_WHITELIST,
        "powerpc" | "powerpc64" => POWERPC_WHITELIST,
        _ => &[],
    };
    whitelist.iter().map(|m| {
        CStr::from_bytes_with_nul(m.as_bytes()).unwrap()
    }).collect()
}

pub fn print_version() {
    unsafe {
        println!("LLVM version: {}.{}",
                 llvm::LLVMRustVersionMajor(), llvm::LLVMRustVersionMinor());
    }
}

pub fn print_passes() {
    unsafe { llvm::LLVMRustPrintPasses(); }
}

pub fn print(req: PrintRequest, sess: &Session) {
    let tm = create_target_machine(sess);
    unsafe {
        match req {
            PrintRequest::TargetCPUs => llvm::LLVMRustPrintTargetCPUs(tm),
            PrintRequest::TargetFeatures => llvm::LLVMRustPrintTargetFeatures(tm),
            _ => bug!("rustc_trans can't handle print request: {:?}", req),
        }
    }
}
