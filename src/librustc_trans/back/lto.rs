// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::link;
use back::write;
use back::symbol_export::{self, ExportedSymbols};
use rustc::session::{self, config};
use llvm;
use llvm::archive_ro::ArchiveRO;
use llvm::{ModuleRef, TargetMachineRef, True, False};
use rustc::util::common::time;
use rustc::util::common::path2cstr;
use rustc::hir::def_id::LOCAL_CRATE;
use back::write::{ModuleConfig, with_llvm_pmb};

use libc;
use flate;

use std::ffi::CString;
use std::path::Path;

pub fn crate_type_allows_lto(crate_type: config::CrateType) -> bool {
    match crate_type {
        config::CrateTypeExecutable |
        config::CrateTypeStaticlib  |
        config::CrateTypeCdylib     => true,

        config::CrateTypeDylib     |
        config::CrateTypeRlib      |
        config::CrateTypeProcMacro => false,
    }
}

pub fn run(sess: &session::Session,
           llmod: ModuleRef,
           tm: TargetMachineRef,
           exported_symbols: &ExportedSymbols,
           config: &ModuleConfig,
           temp_no_opt_bc_filename: &Path) {
    if sess.opts.cg.prefer_dynamic {
        sess.struct_err("cannot prefer dynamic linking when performing LTO")
            .note("only 'staticlib', 'bin', and 'cdylib' outputs are \
                   supported with LTO")
            .emit();
        sess.abort_if_errors();
    }

    // Make sure we actually can run LTO
    for crate_type in sess.crate_types.borrow().iter() {
        if !crate_type_allows_lto(*crate_type) {
            sess.fatal("lto can only be run for executables, cdylibs and \
                            static library outputs");
        }
    }

    let export_threshold =
        symbol_export::crates_export_threshold(&sess.crate_types.borrow()[..]);

    let symbol_filter = &|&(ref name, level): &(String, _)| {
        if symbol_export::is_below_threshold(level, export_threshold) {
            let mut bytes = Vec::with_capacity(name.len() + 1);
            bytes.extend(name.bytes());
            Some(CString::new(bytes).unwrap())
        } else {
            None
        }
    };

    let mut symbol_white_list: Vec<CString> = exported_symbols
        .exported_symbols(LOCAL_CRATE)
        .iter()
        .filter_map(symbol_filter)
        .collect();

    // For each of our upstream dependencies, find the corresponding rlib and
    // load the bitcode from the archive. Then merge it into the current LLVM
    // module that we've got.
    link::each_linked_rlib(sess, &mut |cnum, path| {
        // `#![no_builtins]` crates don't participate in LTO.
        if sess.cstore.is_no_builtins(cnum) {
            return;
        }

        symbol_white_list.extend(
            exported_symbols.exported_symbols(cnum)
                            .iter()
                            .filter_map(symbol_filter));

        let archive = ArchiveRO::open(&path).expect("wanted an rlib");
        let bytecodes = archive.iter().filter_map(|child| {
            child.ok().and_then(|c| c.name().map(|name| (name, c)))
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
                    write::llvm_err(sess.diagnostic(),
                                    format!("failed to load bc of `{}`",
                                            &name[..]));
                }
            });
        }
    });

    // Internalize everything but the exported symbols of the current module
    let arr: Vec<*const libc::c_char> = symbol_white_list.iter()
                                                         .map(|c| c.as_ptr())
                                                         .collect();
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
        let cstr = path2cstr(temp_no_opt_bc_filename);
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
        let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
        assert!(!pass.is_null());
        llvm::LLVMRustAddPass(pm, pass);

        with_llvm_pmb(llmod, config, &mut |b| {
            llvm::LLVMPassManagerBuilderPopulateLTOPassManager(b, pm,
                /* Internalize = */ False,
                /* RunInliner = */ True);
        });

        let pass = llvm::LLVMRustFindAndCreatePass("verify\0".as_ptr() as *const _);
        assert!(!pass.is_null());
        llvm::LLVMRustAddPass(pm, pass);

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
