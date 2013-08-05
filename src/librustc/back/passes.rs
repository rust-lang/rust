// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;

use driver::session::{OptLevel, No, Less, Aggressive};
use driver::session::{Session};
use lib::llvm::{PassRef, ModuleRef,PassManagerRef,TargetDataRef};
use lib::llvm::llvm;
use lib;

pub struct PassManager {
    priv llpm: PassManagerRef
}

impl Drop for PassManager {
    fn drop(&self) {
        unsafe {
            llvm::LLVMDisposePassManager(self.llpm);
        }
    }
}

impl PassManager {
    pub fn new(td: TargetDataRef) -> PassManager {
        unsafe {
            let pm = PassManager {
                llpm: llvm::LLVMCreatePassManager()
            };
            llvm::LLVMAddTargetData(td, pm.llpm);

            return pm;
        }
    }

    pub fn add_pass(&mut self, pass:PassRef) {
        unsafe {
            llvm::LLVMAddPass(self.llpm, pass);
        }
    }

    pub fn add_pass_from_name(&mut self, name:&str) {
        let pass = create_pass(name).unwrap();
        self.add_pass(pass);
    }

    pub fn run(&self, md:ModuleRef) -> bool {
        unsafe {
            llvm::LLVMRunPassManager(self.llpm, md) == lib::llvm::True
        }
    }
}

pub fn create_standard_passes(level: OptLevel) -> ~[~str] {
    let mut passes = ~[];

    // mostly identical to clang 3.3, all differences are documented with comments

    if level != No {
        passes.push(~"targetlibinfo");
        passes.push(~"no-aa");
        // "tbaa" omitted, we don't emit clang-style type-based alias analysis information
        passes.push(~"basicaa");
        passes.push(~"globalopt");
        passes.push(~"ipsccp");
        passes.push(~"deadargelim");
        passes.push(~"instcombine");
        passes.push(~"simplifycfg");
    }

    passes.push(~"basiccg");

    if level != No {
        passes.push(~"prune-eh");
    }

    passes.push(~"inline-cost");

    if level == No || level == Less {
        passes.push(~"always-inline");
    } else {
        passes.push(~"inline");
    }

    if level != No {
        passes.push(~"functionattrs");
        if level == Aggressive {
            passes.push(~"argpromotion");
        }
        passes.push(~"sroa");
        passes.push(~"domtree");
        passes.push(~"early-cse");
        passes.push(~"lazy-value-info");
        passes.push(~"jump-threading");
        passes.push(~"correlated-propagation");
        passes.push(~"simplifycfg");
        passes.push(~"instcombine");
        passes.push(~"tailcallelim");
        passes.push(~"simplifycfg");
        passes.push(~"reassociate");
        passes.push(~"domtree");
        passes.push(~"loops");
        passes.push(~"loop-simplify");
        passes.push(~"lcssa");
        passes.push(~"loop-rotate");
        passes.push(~"licm");
        passes.push(~"lcssa");
        passes.push(~"loop-unswitch");
        passes.push(~"instcombine");
        passes.push(~"scalar-evolution");
        passes.push(~"loop-simplify");
        passes.push(~"lcssa");
        passes.push(~"indvars");
        passes.push(~"loop-idiom");
        passes.push(~"loop-deletion");
        if level == Aggressive {
            passes.push(~"loop-simplify");
            passes.push(~"lcssa");
            passes.push(~"loop-vectorize");
            passes.push(~"loop-simplify");
            passes.push(~"lcssa");
            passes.push(~"scalar-evolution");
            passes.push(~"loop-simplify");
            passes.push(~"lcssa");
        }
        if level != Less {
            passes.push(~"loop-unroll");
            passes.push(~"memdep");
            passes.push(~"gvn");
        }
        passes.push(~"memdep");
        passes.push(~"memcpyopt");
        passes.push(~"sccp");
        passes.push(~"instcombine");
        passes.push(~"lazy-value-info");
        passes.push(~"jump-threading");
        passes.push(~"correlated-propagation");
        passes.push(~"domtree");
        passes.push(~"memdep");
        passes.push(~"dse");
        passes.push(~"adce");
        passes.push(~"simplifycfg");
        passes.push(~"instcombine");
        // clang does `strip-dead-prototypes` here, since it does not emit them
    }

    // rustc emits dead prototypes, so always ask LLVM to strip them
    passes.push(~"strip-dead-prototypes");

    if level != Less {
        passes.push(~"globaldce");
        passes.push(~"constmerge");
    }

    passes
}

pub fn populate_pass_manager(sess: Session, pm: &mut PassManager, pass_list:&[~str]) {
    for nm in pass_list.iter() {
        match create_pass(*nm) {
            Some(p) => pm.add_pass(p),
            None    => sess.warn(fmt!("Unknown pass %s", *nm))
        }
    }
}

pub fn create_pass(name:&str) -> Option<PassRef> {
    do name.as_c_str |s| {
        unsafe {
            let p = llvm::LLVMCreatePass(s);
            if p.is_null() {
                None
            } else {
                Some(p)
            }
        }
    }
}

pub fn list_passes() {
    io::println("\nAvailable Passes:");

    io::println("\nAnalysis Passes:");
    for &(name, desc) in analysis_passes.iter() {
        printfln!("    %-30s -- %s", name, desc);
    }
    io::println("\nTransformation Passes:");
    for &(name, desc) in transform_passes.iter() {
        printfln!("    %-30s -- %s", name, desc);
    }
    io::println("\nUtility Passes:");
    for &(name, desc) in utility_passes.iter() {
        printfln!("    %-30s -- %s", name, desc);
    }
}

/** Analysis Passes */
pub static analysis_passes : &'static [(&'static str, &'static str)] = &'static [
    ("aa-eval",                         "Exhausive Alias Analysis Precision Evaluator"),
    ("asan",                            "AddressSanitizer"),
    ("basicaa",                         "Basic Alias Analysis"),
    ("basiccg",                         "Basic CallGraph Construction"),
    ("block-freq",                      "Block Frequency Analysis"),
    ("cost-model",                      "Cost Model Analysis"),
    ("count-aa",                        "Count Alias Analysis Query Responses"),
    ("da",                              "Dependence Analysis"),
    ("debug-aa",                        "AA Use Debugger"),
    ("domfrontier",                     "Dominance Frontier Construction"),
    ("domtree",                         "Dominator Tree Construction"),
    ("globalsmodref-aa",                "Simple mod/ref analysis for globals"),
    ("instcount",                       "Count the various types of Instructions"),
    ("intervals",                       "Interval Partition Construction"),
    ("iv-users",                        "Induction Variable Users"),
    ("lazy-value-info",                 "Lazy Value Information Analysis"),
    ("libcall-aa",                      "LibCall Alias Analysis"),
    ("lint",                            "Statically lint-check LLVM IR"),
    ("loops",                           "Natural Loop Information"),
    ("memdep",                          "Memory Dependence Analysis"),
    ("module-debuginfo",                "Decodes module-level debug info"),
    ("profile-estimator",               "Estimate profiling information"),
    ("profile-loader",                  "Load profile information from llvmprof.out"),
    ("profile-verifier",                "Verify profiling information"),
    ("regions",                         "Detect single entry single exit regions"),
    ("scalar-evolution",                "Scalar Evolution Analysis"),
    ("scev-aa",                         "Scalar Evolution-based Alias Analysis"),
    ("tbaa",                            "Type-Based Alias Analysis"),
    ("tsan",                            "ThreadSanitizer"),
];

/** Transformation Passes */
pub static transform_passes : &'static [(&'static str, &'static str)] = &'static [
    ("adce",                            "Aggressive Dead Code Elimination"),
    ("always-inline",                   "Inliner for #[inline] functions"),
    ("argpromotion",                    "Promote 'by reference' arguments to scalars"),
    ("bb-vectorize",                    "Basic-Block Vectorization"),
    ("block-placement",                 "Profile Guided Basic Block Placement"),
    ("bounds-checking",                 "Run-time bounds checking"),
    ("break-crit-edges",                "Break critical edges in CFG"),
    ("codegenprepare",                  "Optimize for code generation"),
    ("constmerge",                      "Merge Duplicate Global Constants"),
    ("constprop",                       "Simple constant propagation"),
    ("correlated-propagation",          "Value Propagation"),
    ("da",                              "Data Layout"),
    ("dce",                             "Dead Code Elimination"),
    ("deadargelim",                     "Dead Argument Elimination"),
    ("die",                             "Dead Instruction Elimination"),
    ("dse",                             "Dead Store Elimination"),
    ("early-cse",                       "Early CSE"),
    ("functionattrs",                   "Deduce function attributes"),
    ("globaldce",                       "Dead Global Elimination"),
    ("globalopt",                       "Global Variable Optimizer"),
    ("gvn",                             "Global Value Numbering"),
    ("indvars",                         "Canonicalize Induction Variables"),
    ("inline",                          "Function Integration/Inlining"),
    ("insert-edge-profiling",           "Insert instrumentation for edge profiling"),
    ("insert-gcov-profiling",           "Insert instrumentation for GCOV profiling"),
    ("insert-optimal-edge-profiling",   "Insert optimal instrumentation for edge profiling"),
    ("instcombine",                     "Combine redundant instructions"),
    ("instsimplify",                    "Remove redundant instructions"),
    ("ipconstprop",                     "Interprocedural constant propagation"),
    ("ipsccp",                          "Interprocedural Sparse Conditional Constant Propagation"),
    ("jump-threading",                  "Jump Threading"),
    ("lcssa",                           "Loop-Closed SSA Form Pass"),
    ("licm",                            "Loop Invariant Code Motion"),
    ("loop-deletion",                   "Delete dead loops"),
    ("loop-extract",                    "Extract loops into new functions"),
    ("loop-extract-single",             "Extract at most one loop into a new function"),
    ("loop-idiom",                      "Recognise loop idioms"),
    ("loop-instsimplify",               "Simplify instructions in loops"),
    ("loop-reduce",                     "Loop Strength Reduction"),
    ("loop-rotate",                     "Rotate Loops"),
    ("loop-simplify",                   "Canonicalize natural loops"),
    ("loop-unroll",                     "Unroll loops"),
    ("loop-unswitch",                   "Unswitch loops"),
    ("loop-vectorize",                  "Loop Vectorization"),
    ("lower-expect",                    "Lower 'expect' Intrinsics"),
    ("mem2reg",                         "Promote Memory to Register"),
    ("memcpyopt",                       "MemCpy Optimization"),
    ("mergefunc",                       "Merge Functions"),
    ("mergereturn",                     "Unify function exit nodes"),
    ("partial-inliner",                 "Partial Inliner"),
    ("prune-eh",                        "Remove unused exception handling info"),
    ("reassociate",                     "Reassociate expressions"),
    ("reg2mem",                         "Demote all values to stack slots"),
    ("scalarrepl",                      "Scalar Replacement of Aggregates (DT)"),
    ("scalarrepl-ssa",                  "Scalar Replacement of Aggregates (SSAUp)"),
    ("sccp",                            "Sparse Conditional Constant Propagation"),
    ("simplify-libcalls",               "Simplify well-known library calls"),
    ("simplifycfg",                     "Simplify the CFG"),
    ("sink",                            "Code sinking"),
    ("strip",                           "Strip all symbols from a module"),
    ("strip-dead-debug-info",           "Strip debug info for unused symbols"),
    ("strip-dead-prototypes",           "Strip Unused Function Prototypes"),
    ("strip-debug-declare",             "Strip all llvm.dbg.declare intrinsics"),
    ("strip-nondebug",                  "Strip all symbols, except dbg symbols, from a module"),
    ("sroa",                            "Scalar Replacement of Aggregates"),
    ("tailcallelim",                    "Tail Call Elimination"),
];

/** Utility Passes */
static utility_passes : &'static [(&'static str, &'static str)] = &'static [
    ("instnamer",                       "Assign names to anonymous instructions"),
    ("verify",                          "Module Verifier"),
];

#[test]
fn passes_exist() {
    let mut failed = ~[];
    unsafe { llvm::LLVMInitializePasses(); }
    for &(name,_) in analysis_passes.iter() {
        let pass = create_pass(name);
        if !pass.is_some() {
            failed.push(name);
        } else {
            unsafe { llvm::LLVMDestroyPass(pass.unwrap()) }
        }
    }
    for &(name,_) in transform_passes.iter() {
        let pass = create_pass(name);
        if !pass.is_some() {
            failed.push(name);
        } else {
            unsafe { llvm::LLVMDestroyPass(pass.unwrap()) }
        }
    }
    for &(name,_) in utility_passes.iter() {
        let pass = create_pass(name);
        if !pass.is_some() {
            failed.push(name);
        } else {
            unsafe { llvm::LLVMDestroyPass(pass.unwrap()) }
        }
    }

    if failed.len() > 0 {
        io::println("Some passes don't exist:");
        for &n in failed.iter() {
            printfln!("    %s", n);
        }
        fail!();
    }
}
