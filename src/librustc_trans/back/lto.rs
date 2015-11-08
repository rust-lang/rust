// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::link;
use super::write;
use rustc::session::{self, config};
use llvm;
use llvm::archive_ro::ArchiveRO;
use llvm::{ModuleRef, TargetMachineRef, True, False};
use rustc::util::common::time;
use rustc::util::common::path2cstr;
use back::write::{ModuleConfig, with_llvm_pmb};

use libc;
use flate;

use std::ffi::CString;

pub fn run(sess: &session::Session, llmod: ModuleRef,
           tm: TargetMachineRef, reachable: &[String],
           config: &ModuleConfig,
           name_extra: &str,
           output_names: &config::OutputFilenames) {
    if sess.opts.cg.prefer_dynamic {
        sess.err("cannot prefer dynamic linking when performing LTO");
        sess.note("only 'staticlib' and 'bin' outputs are supported with LTO");
        sess.abort_if_errors();
    }

    // Make sure we actually can run LTO
    for crate_type in sess.crate_types.borrow().iter() {
        match *crate_type {
            config::CrateTypeExecutable | config::CrateTypeStaticlib => {}
            _ => {
                sess.fatal("lto can only be run for executables and \
                            static library outputs");
            }
        }
    }

    // For each of our upstream dependencies, find the corresponding rlib and
    // load the bitcode from the archive. Then merge it into the current LLVM
    // module that we've got.
    link::each_linked_rlib(sess, &mut |_, path| {
        let archive = ArchiveRO::open(&path).expect("wanted an rlib");
        let bytecodes = archive.iter().filter_map(|child| {
            child.name().map(|name| (name, child))
        }).filter(|&(name, _)| name.ends_with("bytecode.deflate"));
        for (name, data) in bytecodes {
            let bc_encoded = data.data();

            let bc_decoded = if is_versioned_bytecode_format(bc_encoded) {
                time(sess.time_passes(), &format!("decode {}", name), || {
                    // Read the version
                    let version = extract_bytecode_format_version(bc_encoded);

                    if version == 1 {
                        // The only version existing so far
                        let data_size = extract_compressed_bytecode_size_v1(bc_encoded);
                        let compressed_data = &bc_encoded[
                            link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET..
                            (link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET + data_size as usize)];

                        match flate::inflate_bytes(compressed_data) {
                            Ok(inflated) => inflated,
                            Err(_) => {
                                sess.fatal(&format!("failed to decompress bc of `{}`",
                                                   name))
                            }
                        }
                    } else {
                        sess.fatal(&format!("Unsupported bytecode format version {}",
                                           version))
                    }
                })
            } else {
                time(sess.time_passes(), &format!("decode {}", name), || {
                    // the object must be in the old, pre-versioning format, so
                    // simply inflate everything and let LLVM decide if it can
                    // make sense of it
                    match flate::inflate_bytes(bc_encoded) {
                        Ok(bc) => bc,
                        Err(_) => {
                            sess.fatal(&format!("failed to decompress bc of `{}`",
                                               name))
                        }
                    }
                })
            };

            let ptr = bc_decoded.as_ptr();
            debug!("linking {}", name);
            time(sess.time_passes(), &format!("ll link {}", name), || unsafe {
                if !llvm::LLVMRustLinkInExternalBitcode(llmod,
                                                        ptr as *const libc::c_char,
                                                        bc_decoded.len() as libc::size_t) {
                    write::llvm_err(sess.diagnostic().handler(),
                                    format!("failed to load bc of `{}`",
                                            &name[..]));
                }
            });
        }
    });

    // Internalize everything but the reachable symbols of the current module
    let cstrs: Vec<CString> = reachable.iter().map(|s| {
        CString::new(s.clone()).unwrap()
    }).collect();
    let arr: Vec<*const libc::c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
    let ptr = arr.as_ptr();
    unsafe {
        llvm::LLVMRustRunRestrictionPass(llmod,
                                         ptr as *const *const libc::c_char,
                                         arr.len() as libc::size_t);
    }

    if sess.no_landing_pads() {
        unsafe {
            llvm::LLVMRustMarkAllFunctionsNounwind(llmod);
        }
    }

    if sess.opts.cg.save_temps {
        let path = output_names.with_extension(&format!("{}.no-opt.lto.bc", name_extra));
        let cstr = path2cstr(&path);
        unsafe {
            llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
        }
    }

    // Now we have one massive module inside of llmod. Time to run the
    // LTO-specific optimization passes that LLVM provides.
    //
    // This code is based off the code found in llvm's LTO code generator:
    //      tools/lto/LTOCodeGenerator.cpp
    debug!("running the pass manager");
    unsafe {
        let pm = llvm::LLVMCreatePassManager();
        llvm::LLVMRustAddAnalysisPasses(tm, pm, llmod);
        llvm::LLVMRustAddPass(pm, "verify\0".as_ptr() as *const _);

        with_llvm_pmb(llmod, config, &mut |b| {
            llvm::LLVMPassManagerBuilderPopulateLTOPassManager(b, pm,
                /* Internalize = */ False,
                /* RunInliner = */ True);
        });

        llvm::LLVMRustAddPass(pm, "verify\0".as_ptr() as *const _);

        time(sess.time_passes(), "LTO passes", ||
             llvm::LLVMRunPassManager(pm, llmod));

        llvm::LLVMDisposePassManager(pm);
    }
    debug!("lto done");
}

fn is_versioned_bytecode_format(bc: &[u8]) -> bool {
    let magic_id_byte_count = link::RLIB_BYTECODE_OBJECT_MAGIC.len();
    return bc.len() > magic_id_byte_count &&
           &bc[..magic_id_byte_count] == link::RLIB_BYTECODE_OBJECT_MAGIC;
}

fn extract_bytecode_format_version(bc: &[u8]) -> u32 {
    let pos = link::RLIB_BYTECODE_OBJECT_VERSION_OFFSET;
    let byte_data = &bc[pos..pos + 4];
    let data = unsafe { *(byte_data.as_ptr() as *const u32) };
    u32::from_le(data)
}

fn extract_compressed_bytecode_size_v1(bc: &[u8]) -> u64 {
    let pos = link::RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET;
    let byte_data = &bc[pos..pos + 8];
    let data = unsafe { *(byte_data.as_ptr() as *const u64) };
    u64::from_le(data)
}
