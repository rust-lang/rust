// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::lto;
use back::link::get_cc_prog;
use driver::driver::{CrateTranslation, OutputFilenames};
use driver::config::NoDebugInfo;
use driver::session::Session;
use driver::config;
use llvm;
use llvm::{ModuleRef, TargetMachineRef, PassManagerRef};
use util::common::time;
use syntax::abi;

use std::c_str::{ToCStr, CString};
use std::io::Command;
use std::ptr;
use std::str;
use libc::{c_uint, c_int};


#[deriving(Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum OutputType {
    OutputTypeBitcode,
    OutputTypeAssembly,
    OutputTypeLlvmAssembly,
    OutputTypeObject,
    OutputTypeExe,
}


pub fn llvm_err(sess: &Session, msg: String) -> ! {
    unsafe {
        let cstr = llvm::LLVMRustGetLastError();
        if cstr == ptr::null() {
            sess.fatal(msg.as_slice());
        } else {
            let err = CString::new(cstr, true);
            let err = String::from_utf8_lossy(err.as_bytes());
            sess.fatal(format!("{}: {}",
                               msg.as_slice(),
                               err.as_slice()).as_slice());
        }
    }
}

pub fn write_output_file(
        sess: &Session,
        target: llvm::TargetMachineRef,
        pm: llvm::PassManagerRef,
        m: ModuleRef,
        output: &Path,
        file_type: llvm::FileType) {
    unsafe {
        output.with_c_str(|output| {
            let result = llvm::LLVMRustWriteOutputFile(
                    target, pm, m, output, file_type);
            if !result {
                llvm_err(sess, "could not write output".to_string());
            }
        })
    }
}


// On android, we by default compile for armv7 processors. This enables
// things like double word CAS instructions (rather than emulating them)
// which are *far* more efficient. This is obviously undesirable in some
// cases, so if any sort of target feature is specified we don't append v7
// to the feature list.
//
// On iOS only armv7 and newer are supported. So it is useful to
// get all hardware potential via VFP3 (hardware floating point)
// and NEON (SIMD) instructions supported by LLVM.
// Note that without those flags various linking errors might
// arise as some of intrinsics are converted into function calls
// and nobody provides implementations those functions
fn target_feature<'a>(sess: &'a Session) -> &'a str {
    match sess.targ_cfg.os {
        abi::OsAndroid => {
            if "" == sess.opts.cg.target_feature.as_slice() {
                "+v7"
            } else {
                sess.opts.cg.target_feature.as_slice()
            }
        },
        abi::OsiOS if sess.targ_cfg.arch == abi::Arm => {
            "+v7,+thumb2,+vfp3,+neon"
        },
        _ => sess.opts.cg.target_feature.as_slice()
    }
}

pub fn run_passes(sess: &Session,
                  trans: &CrateTranslation,
                  output_types: &[OutputType],
                  output: &OutputFilenames) {
    let llmod = trans.module;
    let llcx = trans.context;
    unsafe {
        configure_llvm(sess);

        if sess.opts.cg.save_temps {
            output.with_extension("no-opt.bc").with_c_str(|buf| {
                llvm::LLVMWriteBitcodeToFile(llmod, buf);
            })
        }

        let opt_level = match sess.opts.optimize {
          config::No => llvm::CodeGenLevelNone,
          config::Less => llvm::CodeGenLevelLess,
          config::Default => llvm::CodeGenLevelDefault,
          config::Aggressive => llvm::CodeGenLevelAggressive,
        };
        let use_softfp = sess.opts.cg.soft_float;

        // FIXME: #11906: Omitting frame pointers breaks retrieving the value of a parameter.
        // FIXME: #11954: mac64 unwinding may not work with fp elim
        let no_fp_elim = (sess.opts.debuginfo != NoDebugInfo) ||
                         (sess.targ_cfg.os == abi::OsMacos &&
                          sess.targ_cfg.arch == abi::X86_64);

        // OSX has -dead_strip, which doesn't rely on ffunction_sections
        // FIXME(#13846) this should be enabled for windows
        let ffunction_sections = sess.targ_cfg.os != abi::OsMacos &&
                                 sess.targ_cfg.os != abi::OsWindows;
        let fdata_sections = ffunction_sections;

        let reloc_model = match sess.opts.cg.relocation_model.as_slice() {
            "pic" => llvm::RelocPIC,
            "static" => llvm::RelocStatic,
            "default" => llvm::RelocDefault,
            "dynamic-no-pic" => llvm::RelocDynamicNoPic,
            _ => {
                sess.err(format!("{} is not a valid relocation mode",
                                 sess.opts
                                     .cg
                                     .relocation_model).as_slice());
                sess.abort_if_errors();
                return;
            }
        };

        let code_model = match sess.opts.cg.code_model.as_slice() {
            "default" => llvm::CodeModelDefault,
            "small" => llvm::CodeModelSmall,
            "kernel" => llvm::CodeModelKernel,
            "medium" => llvm::CodeModelMedium,
            "large" => llvm::CodeModelLarge,
            _ => {
                sess.err(format!("{} is not a valid code model",
                                 sess.opts
                                     .cg
                                     .code_model).as_slice());
                sess.abort_if_errors();
                return;
            }
        };

        let tm = sess.targ_cfg
                     .target_strs
                     .target_triple
                     .as_slice()
                     .with_c_str(|t| {
            sess.opts.cg.target_cpu.as_slice().with_c_str(|cpu| {
                target_feature(sess).with_c_str(|features| {
                    llvm::LLVMRustCreateTargetMachine(
                        t, cpu, features,
                        code_model,
                        reloc_model,
                        opt_level,
                        true /* EnableSegstk */,
                        use_softfp,
                        no_fp_elim,
                        ffunction_sections,
                        fdata_sections,
                    )
                })
            })
        });

        // Create the two optimizing pass managers. These mirror what clang
        // does, and are by populated by LLVM's default PassManagerBuilder.
        // Each manager has a different set of passes, but they also share
        // some common passes.
        let fpm = llvm::LLVMCreateFunctionPassManagerForModule(llmod);
        let mpm = llvm::LLVMCreatePassManager();

        // If we're verifying or linting, add them to the function pass
        // manager.
        let addpass = |pass: &str| {
            pass.as_slice().with_c_str(|s| llvm::LLVMRustAddPass(fpm, s))
        };
        if !sess.no_verify() { assert!(addpass("verify")); }

        if !sess.opts.cg.no_prepopulate_passes {
            llvm::LLVMRustAddAnalysisPasses(tm, fpm, llmod);
            llvm::LLVMRustAddAnalysisPasses(tm, mpm, llmod);
            populate_llvm_passes(fpm, mpm, llmod, opt_level,
                                 trans.no_builtins);
        }

        for pass in sess.opts.cg.passes.iter() {
            pass.as_slice().with_c_str(|s| {
                if !llvm::LLVMRustAddPass(mpm, s) {
                    sess.warn(format!("unknown pass {}, ignoring",
                                      *pass).as_slice());
                }
            })
        }

        // Finally, run the actual optimization passes
        time(sess.time_passes(), "llvm function passes", (), |()|
             llvm::LLVMRustRunFunctionPassManager(fpm, llmod));
        time(sess.time_passes(), "llvm module passes", (), |()|
             llvm::LLVMRunPassManager(mpm, llmod));

        // Deallocate managers that we're now done with
        llvm::LLVMDisposePassManager(fpm);
        llvm::LLVMDisposePassManager(mpm);

        // Emit the bytecode if we're either saving our temporaries or
        // emitting an rlib. Whenever an rlib is created, the bytecode is
        // inserted into the archive in order to allow LTO against it.
        if sess.opts.cg.save_temps ||
           (sess.crate_types.borrow().contains(&config::CrateTypeRlib) &&
            sess.opts.output_types.contains(&OutputTypeExe)) {
            output.temp_path(OutputTypeBitcode).with_c_str(|buf| {
                llvm::LLVMWriteBitcodeToFile(llmod, buf);
            })
        }

        if sess.lto() {
            time(sess.time_passes(), "all lto passes", (), |()|
                 lto::run(sess, llmod, tm, trans.reachable.as_slice()));

            if sess.opts.cg.save_temps {
                output.with_extension("lto.bc").with_c_str(|buf| {
                    llvm::LLVMWriteBitcodeToFile(llmod, buf);
                })
            }
        }

        // A codegen-specific pass manager is used to generate object
        // files for an LLVM module.
        //
        // Apparently each of these pass managers is a one-shot kind of
        // thing, so we create a new one for each type of output. The
        // pass manager passed to the closure should be ensured to not
        // escape the closure itself, and the manager should only be
        // used once.
        fn with_codegen(tm: TargetMachineRef, llmod: ModuleRef,
                        no_builtins: bool, f: |PassManagerRef|) {
            unsafe {
                let cpm = llvm::LLVMCreatePassManager();
                llvm::LLVMRustAddAnalysisPasses(tm, cpm, llmod);
                llvm::LLVMRustAddLibraryInfo(cpm, llmod, no_builtins);
                f(cpm);
                llvm::LLVMDisposePassManager(cpm);
            }
        }

        let mut object_file = None;
        let mut needs_metadata = false;
        for output_type in output_types.iter() {
            let path = output.path(*output_type);
            match *output_type {
                OutputTypeBitcode => {
                    path.with_c_str(|buf| {
                        llvm::LLVMWriteBitcodeToFile(llmod, buf);
                    })
                }
                OutputTypeLlvmAssembly => {
                    path.with_c_str(|output| {
                        with_codegen(tm, llmod, trans.no_builtins, |cpm| {
                            llvm::LLVMRustPrintModule(cpm, llmod, output);
                        })
                    })
                }
                OutputTypeAssembly => {
                    // If we're not using the LLVM assembler, this function
                    // could be invoked specially with output_type_assembly,
                    // so in this case we still want the metadata object
                    // file.
                    let ty = OutputTypeAssembly;
                    let path = if sess.opts.output_types.contains(&ty) {
                       path
                    } else {
                        needs_metadata = true;
                        output.temp_path(OutputTypeAssembly)
                    };
                    with_codegen(tm, llmod, trans.no_builtins, |cpm| {
                        write_output_file(sess, tm, cpm, llmod, &path,
                                        llvm::AssemblyFile);
                    });
                }
                OutputTypeObject => {
                    object_file = Some(path);
                }
                OutputTypeExe => {
                    object_file = Some(output.temp_path(OutputTypeObject));
                    needs_metadata = true;
                }
            }
        }

        time(sess.time_passes(), "codegen passes", (), |()| {
            match object_file {
                Some(ref path) => {
                    with_codegen(tm, llmod, trans.no_builtins, |cpm| {
                        write_output_file(sess, tm, cpm, llmod, path,
                                        llvm::ObjectFile);
                    });
                }
                None => {}
            }
            if needs_metadata {
                with_codegen(tm, trans.metadata_module,
                             trans.no_builtins, |cpm| {
                    let out = output.temp_path(OutputTypeObject)
                                    .with_extension("metadata.o");
                    write_output_file(sess, tm, cpm,
                                    trans.metadata_module, &out,
                                    llvm::ObjectFile);
                })
            }
        });

        llvm::LLVMRustDisposeTargetMachine(tm);
        llvm::LLVMDisposeModule(trans.metadata_module);
        llvm::LLVMDisposeModule(llmod);
        llvm::LLVMContextDispose(llcx);
        if sess.time_llvm_passes() { llvm::LLVMRustPrintPassTimings(); }
    }
}

pub fn run_assembler(sess: &Session, outputs: &OutputFilenames) {
    let pname = get_cc_prog(sess);
    let mut cmd = Command::new(pname.as_slice());

    cmd.arg("-c").arg("-o").arg(outputs.path(OutputTypeObject))
                           .arg(outputs.temp_path(OutputTypeAssembly));
    debug!("{}", &cmd);

    match cmd.output() {
        Ok(prog) => {
            if !prog.status.success() {
                sess.err(format!("linking with `{}` failed: {}",
                                 pname,
                                 prog.status).as_slice());
                sess.note(format!("{}", &cmd).as_slice());
                let mut note = prog.error.clone();
                note.push_all(prog.output.as_slice());
                sess.note(str::from_utf8(note.as_slice()).unwrap());
                sess.abort_if_errors();
            }
        },
        Err(e) => {
            sess.err(format!("could not exec the linker `{}`: {}",
                             pname,
                             e).as_slice());
            sess.abort_if_errors();
        }
    }
}

unsafe fn configure_llvm(sess: &Session) {
    use std::sync::{Once, ONCE_INIT};
    static mut INIT: Once = ONCE_INIT;

    // Copy what clang does by turning on loop vectorization at O2 and
    // slp vectorization at O3
    let vectorize_loop = !sess.opts.cg.no_vectorize_loops &&
                         (sess.opts.optimize == config::Default ||
                          sess.opts.optimize == config::Aggressive);
    let vectorize_slp = !sess.opts.cg.no_vectorize_slp &&
                        sess.opts.optimize == config::Aggressive;

    let mut llvm_c_strs = Vec::new();
    let mut llvm_args = Vec::new();
    {
        let add = |arg: &str| {
            let s = arg.to_c_str();
            llvm_args.push(s.as_ptr());
            llvm_c_strs.push(s);
        };
        add("rustc"); // fake program name
        if vectorize_loop { add("-vectorize-loops"); }
        if vectorize_slp  { add("-vectorize-slp");   }
        if sess.time_llvm_passes() { add("-time-passes"); }
        if sess.print_llvm_passes() { add("-debug-pass=Structure"); }

        for arg in sess.opts.cg.llvm_args.iter() {
            add((*arg).as_slice());
        }
    }

    INIT.doit(|| {
        llvm::LLVMInitializePasses();

        // Only initialize the platforms supported by Rust here, because
        // using --llvm-root will have multiple platforms that rustllvm
        // doesn't actually link to and it's pointless to put target info
        // into the registry that Rust cannot generate machine code for.
        llvm::LLVMInitializeX86TargetInfo();
        llvm::LLVMInitializeX86Target();
        llvm::LLVMInitializeX86TargetMC();
        llvm::LLVMInitializeX86AsmPrinter();
        llvm::LLVMInitializeX86AsmParser();

        llvm::LLVMInitializeARMTargetInfo();
        llvm::LLVMInitializeARMTarget();
        llvm::LLVMInitializeARMTargetMC();
        llvm::LLVMInitializeARMAsmPrinter();
        llvm::LLVMInitializeARMAsmParser();

        llvm::LLVMInitializeMipsTargetInfo();
        llvm::LLVMInitializeMipsTarget();
        llvm::LLVMInitializeMipsTargetMC();
        llvm::LLVMInitializeMipsAsmPrinter();
        llvm::LLVMInitializeMipsAsmParser();

        llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int,
                                     llvm_args.as_ptr());
    });
}

unsafe fn populate_llvm_passes(fpm: llvm::PassManagerRef,
                               mpm: llvm::PassManagerRef,
                               llmod: ModuleRef,
                               opt: llvm::CodeGenOptLevel,
                               no_builtins: bool) {
    // Create the PassManagerBuilder for LLVM. We configure it with
    // reasonable defaults and prepare it to actually populate the pass
    // manager.
    let builder = llvm::LLVMPassManagerBuilderCreate();
    match opt {
        llvm::CodeGenLevelNone => {
            // Don't add lifetime intrinsics at O0
            llvm::LLVMRustAddAlwaysInlinePass(builder, false);
        }
        llvm::CodeGenLevelLess => {
            llvm::LLVMRustAddAlwaysInlinePass(builder, true);
        }
        // numeric values copied from clang
        llvm::CodeGenLevelDefault => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder,
                                                                225);
        }
        llvm::CodeGenLevelAggressive => {
            llvm::LLVMPassManagerBuilderUseInlinerWithThreshold(builder,
                                                                275);
        }
    }
    llvm::LLVMPassManagerBuilderSetOptLevel(builder, opt as c_uint);
    llvm::LLVMRustAddBuilderLibraryInfo(builder, llmod, no_builtins);

    // Use the builder to populate the function/module pass managers.
    llvm::LLVMPassManagerBuilderPopulateFunctionPassManager(builder, fpm);
    llvm::LLVMPassManagerBuilderPopulateModulePassManager(builder, mpm);
    llvm::LLVMPassManagerBuilderDispose(builder);

    match opt {
        llvm::CodeGenLevelDefault | llvm::CodeGenLevelAggressive => {
            "mergefunc".with_c_str(|s| llvm::LLVMRustAddPass(mpm, s));
        }
        _ => {}
    };
}
