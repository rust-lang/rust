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
use libc::{c_int, c_char};
use std::ffi::CString;

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

const X86_WHITELIST: &'static [&'static str] = &["avx\0", "avx2\0", "bmi\0", "bmi2\0", "sse\0",
                                                 "sse2\0", "sse3\0", "sse4.1\0", "sse4.2\0",
                                                 "ssse3\0", "tbm\0", "lzcnt\0", "popcnt\0",
                                                 "sse4a\0", "rdrnd\0", "rdseed\0", "fma\0"];

const HEXAGON_WHITELIST: &'static [&'static str] = &["hvx\0", "hvx-double\0"];

const POWERPC_WHITELIST: &'static [&'static str] = &["altivec\0", "vsx\0"];

pub fn target_features(sess: &Session) -> Vec<Symbol> {
    let target_machine = create_target_machine(sess);

    let whitelist = match &*sess.target.target.arch {
        "arm" => ARM_WHITELIST,
        "x86" | "x86_64" => X86_WHITELIST,
        "hexagon" => HEXAGON_WHITELIST,
        "powerpc" | "powerpc64" => POWERPC_WHITELIST,
        _ => &[],
    };

    let mut features = Vec::new();
    for feat in whitelist {
        assert_eq!(feat.chars().last(), Some('\0'));
        if unsafe { llvm::LLVMRustHasFeature(target_machine, feat.as_ptr() as *const c_char) } {
            features.push(Symbol::intern(&feat[..feat.len() - 1]));
        }
    }
    features
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

pub fn enable_llvm_debug() {
    unsafe { llvm::LLVMRustSetDebug(1); }
}
