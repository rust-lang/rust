// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session;
use lib::llvm::{PassRef, ModuleRef,PassManagerRef,TargetDataRef};
use lib::llvm::llvm;
use lib;

pub struct PassManager {
    priv llpm: PassManagerRef
}

impl Drop for PassManager {
    fn finalize(&self) {
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

    pub fn addPass(&mut self, pass:PassRef) {
        unsafe {
            llvm::LLVMAddPass(self.llpm, pass);
        }
    }

    pub fn run(&self, md:ModuleRef) -> bool {
        unsafe {
            llvm::LLVMRunPassManager(self.llpm, md) == lib::llvm::True
        }
    }
}


pub fn populatePassManager(pm: &mut PassManager, level:session::OptLevel) {
    unsafe {
        // We add a lot of normally-unused prototypes, so always strip them
        // straight away, later passes will get rid of any that are optimized
        // away
        pm.addPass(llvm::LLVMCreateStripDeadPrototypesPass());
        if level == session::No {
            pm.addPass(llvm::LLVMCreateAlwaysInlinerPass());

            return;
        }

        //NOTE: Add library info

        pm.addPass(llvm::LLVMCreateTypeBasedAliasAnalysisPass());
        pm.addPass(llvm::LLVMCreateBasicAliasAnalysisPass());

        pm.addPass(llvm::LLVMCreateSROAPass());
        pm.addPass(llvm::LLVMCreateEarlyCSEPass());
        pm.addPass(llvm::LLVMCreateLowerExpectIntrinsicPass());

        pm.addPass(llvm::LLVMCreateGlobalOptimizerPass());
        pm.addPass(llvm::LLVMCreateIPSCCPPass());
        pm.addPass(llvm::LLVMCreateDeadArgEliminationPass());
        pm.addPass(llvm::LLVMCreateInstructionCombiningPass());
        pm.addPass(llvm::LLVMCreateCFGSimplificationPass());

        pm.addPass(llvm::LLVMCreatePruneEHPass());

        if level == session::Aggressive {
            // Do this before inlining, since inlining might
            // make minor changes to functions that mean they
            // can't be merged, despite being almost identical
            pm.addPass(llvm::LLVMCreateMergeFunctionsPass());
        }

        match level {
            session::Less       => pm.addPass(llvm::LLVMCreateFunctionInliningPass(200)),
            session::Default    => pm.addPass(llvm::LLVMCreateFunctionInliningPass(225)),
            session::Aggressive => pm.addPass(llvm::LLVMCreateFunctionInliningPass(275)),
            session::No         => ()
        }

        pm.addPass(llvm::LLVMCreateFunctionAttrsPass());

        if level == session::Aggressive {
            pm.addPass(llvm::LLVMCreateArgumentPromotionPass());
        }

        pm.addPass(llvm::LLVMCreateEarlyCSEPass());
        pm.addPass(llvm::LLVMCreateSimplifyLibCallsPass());
        pm.addPass(llvm::LLVMCreateJumpThreadingPass());
        pm.addPass(llvm::LLVMCreateCorrelatedValuePropagationPass());
        pm.addPass(llvm::LLVMCreateCFGSimplificationPass());
        pm.addPass(llvm::LLVMCreateInstructionCombiningPass());

        pm.addPass(llvm::LLVMCreateTailCallEliminationPass());
        pm.addPass(llvm::LLVMCreateCFGSimplificationPass());
        pm.addPass(llvm::LLVMCreateReassociatePass());
        pm.addPass(llvm::LLVMCreateLoopRotatePass());
        pm.addPass(llvm::LLVMCreateLICMPass());

        pm.addPass(llvm::LLVMCreateInstructionCombiningPass());
        pm.addPass(llvm::LLVMCreateIndVarSimplifyPass());
        pm.addPass(llvm::LLVMCreateLoopIdiomPass());
        pm.addPass(llvm::LLVMCreateLoopDeletionPass());

        if level == session::Aggressive {
            pm.addPass(llvm::LLVMCreateLoopVectorizePass());
        }
        pm.addPass(llvm::LLVMCreateLoopUnrollPass());

        if level != session::Less {
            pm.addPass(llvm::LLVMCreateGVNPass());
        }
        pm.addPass(llvm::LLVMCreateMemCpyOptPass());
        pm.addPass(llvm::LLVMCreateSCCPPass());

        pm.addPass(llvm::LLVMCreateInstructionCombiningPass());
        pm.addPass(llvm::LLVMCreateJumpThreadingPass());
        pm.addPass(llvm::LLVMCreateCorrelatedValuePropagationPass());
        pm.addPass(llvm::LLVMCreateDeadStoreEliminationPass());

        pm.addPass(llvm::LLVMCreateBBVectorizePass());
        pm.addPass(llvm::LLVMCreateInstructionCombiningPass());
        pm.addPass(llvm::LLVMCreateEarlyCSEPass());

        pm.addPass(llvm::LLVMCreateLoopUnrollPass());

        pm.addPass(llvm::LLVMCreateAggressiveDCEPass());
        pm.addPass(llvm::LLVMCreateCFGSimplificationPass());
        pm.addPass(llvm::LLVMCreateInstructionSimplifierPass());

        if level != session::Less {
            pm.addPass(llvm::LLVMCreateGlobalDCEPass());
            pm.addPass(llvm::LLVMCreateConstantMergePass());
        }
    }
}
