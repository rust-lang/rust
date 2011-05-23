import driver::session;
import lib::llvm::llvm;
import middle::trans;
import std::str;
import std::fs;

import lib::llvm::llvm::ModuleRef;
import lib::llvm::llvm::ValueRef;
import lib::llvm::mk_pass_manager;
import lib::llvm::mk_target_data;
import lib::llvm::mk_type_names;
import lib::llvm::False;
import lib::llvm::True;

tag output_type {
    output_type_none;
    output_type_bitcode;
    output_type_assembly;
    output_type_object;
    output_type_exe;
}

fn llvm_err(session::session sess, str msg) {
    auto buf = llvm::LLVMRustGetLastError();
    if ((buf as uint) == 0u) {
        sess.err(msg);
    } else {
        sess.err(msg + ": " + str::str_from_cstr(buf));
    }
    fail;
}

fn link_intrinsics(session::session sess, ModuleRef llmod) {
    auto path = fs::connect(sess.get_opts().sysroot, "intrinsics.bc");
    auto membuf =
        llvm::LLVMRustCreateMemoryBufferWithContentsOfFile(str::buf(path));
    if ((membuf as uint) == 0u) {
        llvm_err(sess, "installation problem: couldn't open intrinstics.bc");
        fail;
    }

    auto llintrinsicsmod = llvm::LLVMRustParseBitcode(membuf);
    llvm::LLVMDisposeMemoryBuffer(membuf);

    if ((llintrinsicsmod as uint) == 0u) {
        llvm_err(sess, "installation problem: couldn't parse intrinstics.bc");
        fail;
    }

    auto linkres = llvm::LLVMLinkModules(llmod, llintrinsicsmod);
    llvm::LLVMDisposeModule(llintrinsicsmod);
    
    if (linkres == False) {
        llvm_err(sess, "couldn't link the module with the intrinsics");
        fail;
    }
}

mod write {
    fn is_object_or_assembly_or_exe(output_type ot) -> bool {
        if ( (ot == output_type_assembly) || 
             (ot == output_type_object) ||
             (ot == output_type_exe) ) {
            ret true;
        }
        ret false;
    }

    // Decides what to call an intermediate file, given the name of the output
    // and the extension to use.
    fn mk_intermediate_name(str output_path, str extension) -> str {
        auto dot_pos = str::index(output_path, '.' as u8);
        auto stem;
        if (dot_pos < 0) {
            stem = output_path;
        } else {
            stem = str::substr(output_path, 0u, dot_pos as uint);
        }
        ret stem + "." + extension;
    }

    fn run_passes(session::session sess, ModuleRef llmod, str output) {

        auto opts = sess.get_opts();

        if (opts.time_llvm_passes) {
          llvm::LLVMRustEnableTimePasses();
        }

        link_intrinsics(sess, llmod);

        auto pm = mk_pass_manager();
        auto td = mk_target_data(x86::get_data_layout());
        llvm::LLVMAddTargetData(td.lltd, pm.llpm);

        // TODO: run the linter here also, once there are llvm-c bindings for
        // it.

        // Generate a pre-optimization intermediate file if -save-temps was
        // specified.
        if (opts.save_temps) {
            alt (opts.output_type) {
                case (output_type_bitcode) {
                    if (opts.optimize != 0u) {
                        auto filename = mk_intermediate_name(output,
                                                             "no-opt.bc");
                        llvm::LLVMWriteBitcodeToFile(llmod,
                                                    str::buf(filename));
                    }
                }
                case (_) {
                    auto filename = mk_intermediate_name(output, "bc");
                    llvm::LLVMWriteBitcodeToFile(llmod, str::buf(filename));
                }
            }
        }

        if (opts.verify) {
            llvm::LLVMAddVerifierPass(pm.llpm);
        }

        // FIXME: This is mostly a copy of the bits of opt's -O2 that are
        // available in the C api.
        // FIXME2: We might want to add optimization levels like -O1, -O2,
        // -Os, etc
        // FIXME3: Should we expose and use the pass lists used by the opt
        // tool?
        if (opts.optimize != 0u) {
            auto fpm = mk_pass_manager();
            llvm::LLVMAddTargetData(td.lltd, fpm.llpm);
            llvm::LLVMAddStandardFunctionPasses(fpm.llpm, 2u);
            llvm::LLVMRunPassManager(fpm.llpm, llmod);

            let uint threshold = 225u;
            if (opts.optimize == 3u) {
                threshold = 275u;
            }

            llvm::LLVMAddStandardModulePasses(pm.llpm,
                                              // optimization level
                                              opts.optimize,
                                              False, // optimize for size
                                              True,  // unit-at-a-time
                                              True,  // unroll loops
                                              True,  // simplify lib calls
                                              True,  // have exceptions
                                              threshold); // inline threshold
        }

        if (opts.verify) {
            llvm::LLVMAddVerifierPass(pm.llpm);
        }

        if (is_object_or_assembly_or_exe(opts.output_type)) {
            let int LLVMAssemblyFile = 0;
            let int LLVMObjectFile = 1;
            let int LLVMNullFile = 2;
            auto FileType;
            if ((opts.output_type == output_type_object) ||
                (opts.output_type == output_type_exe)) {
                FileType = LLVMObjectFile;
            } else {
                FileType = LLVMAssemblyFile;
            }

            // Write optimized bitcode if --save-temps was on.
            if (opts.save_temps) {

                // Always output the bitcode file with --save-temps
                auto filename = mk_intermediate_name(output, "opt.bc");
                llvm::LLVMRunPassManager(pm.llpm, llmod);
                llvm::LLVMWriteBitcodeToFile(llmod, str::buf(filename));
                pm = mk_pass_manager();

                // Save the assembly file if -S is used
                if (opts.output_type == output_type_assembly) {
                        llvm::LLVMRustWriteOutputFile(pm.llpm, llmod,
                               str::buf(x86::get_target_triple()),
                               str::buf(output), LLVMAssemblyFile);
                }

                // Save the object file for -c or --save-temps alone
                // This .o is needed when an exe is built
                if ((opts.output_type == output_type_object) ||
                    (opts.output_type == output_type_exe)) {
                        llvm::LLVMRustWriteOutputFile(pm.llpm, llmod,
                               str::buf(x86::get_target_triple()),
                               str::buf(output), LLVMObjectFile);
               }
            } else {

                // If we aren't saving temps then just output the file
                // type corresponding to the '-c' or '-S' flag used
                llvm::LLVMRustWriteOutputFile(pm.llpm, llmod,
                                     str::buf(x86::get_target_triple()),
                                     str::buf(output),
                                     FileType);
            }

            // Clean up and return
            llvm::LLVMDisposeModule(llmod);
            if (opts.time_llvm_passes) {
              llvm::LLVMRustPrintPassTimings();
            }
            ret;
        }

        // If only a bitcode file is asked for by using the '--emit-llvm'
        // flag, then output it here
        llvm::LLVMRunPassManager(pm.llpm, llmod);

        llvm::LLVMWriteBitcodeToFile(llmod, str::buf(output));
        llvm::LLVMDisposeModule(llmod);

        if (opts.time_llvm_passes) {
          llvm::LLVMRustPrintPassTimings();
        }
    }
}

