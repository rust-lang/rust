import driver.session;
import lib.llvm.llvm;
import middle.trans;
import std.Str;
import std.FS;

import lib.llvm.llvm.ModuleRef;
import lib.llvm.llvm.ValueRef;
import lib.llvm.mk_pass_manager;
import lib.llvm.mk_target_data;
import lib.llvm.mk_type_names;
import lib.llvm.False;

tag output_type {
    output_type_none;
    output_type_bitcode;
    output_type_assembly;
    output_type_object;
}

fn llvm_err(session.session sess, str msg) {
    sess.err(msg + ": " + Str.str_from_cstr(llvm.LLVMRustGetLastError()));
    fail;
}

fn link_intrinsics(session.session sess, ModuleRef llmod) {
    auto path = FS.connect(sess.get_opts().sysroot, "intrinsics.bc");
    auto membuf =
        llvm.LLVMRustCreateMemoryBufferWithContentsOfFile(Str.buf(path));
    if ((membuf as uint) == 0u) {
        llvm_err(sess, "installation problem: couldn't open intrinstics.bc");
        fail;
    }

    auto llintrinsicsmod = llvm.LLVMRustParseBitcode(membuf);
    if ((llintrinsicsmod as uint) == 0u) {
        llvm_err(sess, "installation problem: couldn't parse intrinstics.bc");
        fail;
    }

    if (llvm.LLVMLinkModules(llmod, llintrinsicsmod) == False) {
        llvm_err(sess, "couldn't link the module with the intrinsics");
        fail;
    }
}

mod Write {
    fn is_object_or_assembly(output_type ot) -> bool {
        if (ot == output_type_assembly) {
            ret true;
        }
        if (ot == output_type_object) {
            ret true;
        }
        ret false;
    }

    // Decides what to call an intermediate file, given the name of the output
    // and the extension to use.
    fn mk_intermediate_name(str output_path, str extension) -> str {
        auto dot_pos = Str.index(output_path, '.' as u8);
        auto stem;
        if (dot_pos < 0) {
            stem = output_path;
        } else {
            stem = Str.substr(output_path, 0u, dot_pos as uint);
        }
        ret stem + "." + extension;
    }

    fn run_passes(session.session sess, ModuleRef llmod, str output) {
        link_intrinsics(sess, llmod);

        auto pm = mk_pass_manager();
        auto opts = sess.get_opts();

        // TODO: run the linter here also, once there are llvm-c bindings for
        // it.

        // Generate a pre-optimization intermediate file if -save-temps was
        // specified.
        if (opts.save_temps) {
            alt (opts.output_type) {
                case (output_type_bitcode) {
                    if (opts.optimize) {
                        auto filename = mk_intermediate_name(output,
                                                             "no-opt.bc");
                        llvm.LLVMWriteBitcodeToFile(llmod,
                                                    Str.buf(filename));
                    }
                }
                case (_) {
                    auto filename = mk_intermediate_name(output, "bc");
                    llvm.LLVMWriteBitcodeToFile(llmod, Str.buf(filename));
                }
            }
        }

        // FIXME: This is mostly a copy of the bits of opt's -O2 that are
        // available in the C api.
        // FIXME2: We might want to add optimization levels like -O1, -O2,
        // -Os, etc
        // FIXME3: Should we expose and use the pass lists used by the opt
        // tool?
        if (opts.optimize) {
            auto fpm = mk_pass_manager();

            // createStandardFunctionPasses
            llvm.LLVMAddTypeBasedAliasAnalysisPass(fpm.llpm);
            llvm.LLVMAddBasicAliasAnalysisPass(fpm.llpm);
            llvm.LLVMAddCFGSimplificationPass(fpm.llpm);
            llvm.LLVMAddScalarReplAggregatesPass(fpm.llpm);
            llvm.LLVMAddEarlyCSEPass(fpm.llpm);

            llvm.LLVMRunPassManager(fpm.llpm, llmod);

            // createStandardModulePasses
            llvm.LLVMAddTypeBasedAliasAnalysisPass(pm.llpm);
            llvm.LLVMAddBasicAliasAnalysisPass(pm.llpm);
            llvm.LLVMAddGlobalOptimizerPass(pm.llpm);
            llvm.LLVMAddIPSCCPPass(pm.llpm);
            llvm.LLVMAddDeadArgEliminationPass(pm.llpm);
            llvm.LLVMAddInstructionCombiningPass(pm.llpm);
            llvm.LLVMAddCFGSimplificationPass(pm.llpm);
            llvm.LLVMAddPruneEHPass(pm.llpm);
            llvm.LLVMAddFunctionInliningPass(pm.llpm);
            llvm.LLVMAddFunctionAttrsPass(pm.llpm);
            llvm.LLVMAddScalarReplAggregatesPassSSA(pm.llpm);
            llvm.LLVMAddEarlyCSEPass(pm.llpm);
            llvm.LLVMAddSimplifyLibCallsPass(pm.llpm);
            llvm.LLVMAddJumpThreadingPass(pm.llpm);
            llvm.LLVMAddCorrelatedValuePropagationPass(pm.llpm);
            llvm.LLVMAddCFGSimplificationPass(pm.llpm);
            llvm.LLVMAddInstructionCombiningPass(pm.llpm);
            llvm.LLVMAddTailCallEliminationPass(pm.llpm);
            llvm.LLVMAddCFGSimplificationPass(pm.llpm);
            llvm.LLVMAddReassociatePass(pm.llpm);
            llvm.LLVMAddLoopRotatePass(pm.llpm);
            llvm.LLVMAddLICMPass(pm.llpm);
            llvm.LLVMAddLoopUnswitchPass(pm.llpm);
            llvm.LLVMAddInstructionCombiningPass(pm.llpm);
            llvm.LLVMAddIndVarSimplifyPass(pm.llpm);
            llvm.LLVMAddLoopIdiomPass(pm.llpm);
            llvm.LLVMAddLoopDeletionPass(pm.llpm);
            llvm.LLVMAddLoopUnrollPass(pm.llpm);
            llvm.LLVMAddInstructionCombiningPass(pm.llpm);
            llvm.LLVMAddGVNPass(pm.llpm);
            llvm.LLVMAddMemCpyOptPass(pm.llpm);
            llvm.LLVMAddSCCPPass(pm.llpm);
            llvm.LLVMAddInstructionCombiningPass(pm.llpm);
            llvm.LLVMAddJumpThreadingPass(pm.llpm);
            llvm.LLVMAddCorrelatedValuePropagationPass(pm.llpm);
            llvm.LLVMAddDeadStoreEliminationPass(pm.llpm);
            llvm.LLVMAddAggressiveDCEPass(pm.llpm);
            llvm.LLVMAddCFGSimplificationPass(pm.llpm);
            llvm.LLVMAddStripDeadPrototypesPass(pm.llpm);
            llvm.LLVMAddDeadTypeEliminationPass(pm.llpm);
            llvm.LLVMAddConstantMergePass(pm.llpm);
        }

        if (opts.verify) {
            llvm.LLVMAddVerifierPass(pm.llpm);
        }

        // TODO: Write .s if -c was specified and -save-temps was on.
        if (is_object_or_assembly(opts.output_type)) {
            let int LLVMAssemblyFile = 0;
            let int LLVMObjectFile = 1;
            let int LLVMNullFile = 2;
            auto FileType;
            if (opts.output_type == output_type_object) {
                FileType = LLVMObjectFile;
            } else {
                FileType = LLVMAssemblyFile;
            }

            // Write optimized bitcode if --save-temps was on.
            if (opts.save_temps) {
                alt (opts.output_type) {
                    case (output_type_bitcode) { /* nothing to do */ }
                    case (_) {
                        auto filename = mk_intermediate_name(output,
                                                             "opt.bc");
                        llvm.LLVMRunPassManager(pm.llpm, llmod);
                        llvm.LLVMWriteBitcodeToFile(llmod,
                                                    Str.buf(filename));
                        pm = mk_pass_manager();
                    }
                }
            }

            llvm.LLVMRustWriteOutputFile(pm.llpm, llmod,
                                         Str.buf(x86.get_target_triple()),
                                         Str.buf(output),
                                         FileType);
            llvm.LLVMDisposeModule(llmod);
            ret;
        }

        llvm.LLVMRunPassManager(pm.llpm, llmod);

        llvm.LLVMWriteBitcodeToFile(llmod, Str.buf(output));
        llvm.LLVMDisposeModule(llmod);
    }
}

