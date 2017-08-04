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
use back::symbol_export;
use rustc::session::config;
use errors::{FatalError, Handler};
use llvm;
use llvm::archive_ro::ArchiveRO;
use llvm::{ModuleRef, TargetMachineRef, True, False};
use rustc::util::common::time;
use rustc::util::common::path2cstr;
use rustc::hir::def_id::LOCAL_CRATE;
use back::write::{ModuleConfig, with_llvm_pmb, CodegenContext};

use libc;
use flate2::read::DeflateDecoder;

use std::io::Read;
use std::ffi::CString;
use std::path::Path;
use std::ptr::read_unaligned;

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

pub fn run(cgcx: &CodegenContext,
           diag_handler: &Handler,
           llmod: ModuleRef,
           tm: TargetMachineRef,
           config: &ModuleConfig,
           temp_no_opt_bc_filename: &Path) -> Result<(), FatalError> {
    if cgcx.opts.cg.prefer_dynamic {
        diag_handler.struct_err("cannot prefer dynamic linking when performing LTO")
                    .note("only 'staticlib', 'bin', and 'cdylib' outputs are \
                           supported with LTO")
                    .emit();
        return Err(FatalError)
    }

    // Make sure we actually can run LTO
    for crate_type in cgcx.crate_types.iter() {
        if !crate_type_allows_lto(*crate_type) {
            let e = diag_handler.fatal("lto can only be run for executables, cdylibs and \
                                        static library outputs");
            return Err(e)
        }
    }

    let export_threshold =
        symbol_export::crates_export_threshold(&cgcx.crate_types);

    let symbol_filter = &|&(ref name, _, level): &(String, _, _)| {
        if symbol_export::is_below_threshold(level, export_threshold) {
            let mut bytes = Vec::with_capacity(name.len() + 1);
            bytes.extend(name.bytes());
            Some(CString::new(bytes).unwrap())
        } else {
            None
        }
    };

    let mut symbol_white_list: Vec<CString> = cgcx.exported_symbols
        .exported_symbols(LOCAL_CRATE)
        .iter()
        .filter_map(symbol_filter)
        .collect();

    // For each of our upstream dependencies, find the corresponding rlib and
    // load the bitcode from the archive. Then merge it into the current LLVM
    // module that we've got.
    for &(cnum, ref path) in cgcx.each_linked_rlib_for_lto.iter() {
        symbol_white_list.extend(
            cgcx.exported_symbols.exported_symbols(cnum)
                                 .iter()
                                 .filter_map(symbol_filter));

        let archive = ArchiveRO::open(&path).expect("wanted an rlib");
        let bytecodes = archive.iter().filter_map(|child| {
            child.ok().and_then(|c| c.name().map(|name| (name, c)))
        }).filter(|&(name, _)| name.ends_with("bytecode.deflate"));
        for (name, data) in bytecodes {
            let bc_encoded = data.data();

            let bc_decoded = if is_versioned_bytecode_format(bc_encoded) {
                time(cgcx.time_passes, &format!("decode {}", name), || {
                    // Read the version
                    let version = extract_bytecode_format_version(bc_encoded);

                    if version == 1 {
                        // The only version existing so far
                        let data_size = extract_compressed_bytecode_size_v1(bc_encoded);
                        let compressed_data = &bc_encoded[
                            link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET..
                            (link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET + data_size as usize)];

                        let mut inflated = Vec::new();
                        let res = DeflateDecoder::new(compressed_data)
                            .read_to_end(&mut inflated);
                        if res.is_err() {
                            let msg = format!("failed to decompress bc of `{}`",
                                              name);
                            Err(diag_handler.fatal(&msg))
                        } else {
                            Ok(inflated)
                        }
                    } else {
                        Err(diag_handler.fatal(&format!("Unsupported bytecode format version {}",
                                                        version)))
                    }
                })?
            } else {
                time(cgcx.time_passes, &format!("decode {}", name), || {
                    // the object must be in the old, pre-versioning format, so
                    // simply inflate everything and let LLVM decide if it can
                    // make sense of it
                    let mut inflated = Vec::new();
                    let res = DeflateDecoder::new(bc_encoded)
                        .read_to_end(&mut inflated);
                    if res.is_err() {
                        let msg = format!("failed to decompress bc of `{}`",
                                          name);
                        Err(diag_handler.fatal(&msg))
                    } else {
                        Ok(inflated)
                    }
                })?
            };

            let ptr = bc_decoded.as_ptr();
            debug!("linking {}", name);
            time(cgcx.time_passes, &format!("ll link {}", name), || unsafe {
                if llvm::LLVMRustLinkInExternalBitcode(llmod,
                                                       ptr as *const libc::c_char,
                                                       bc_decoded.len() as libc::size_t) {
                    Ok(())
                } else {
                    let msg = format!("failed to load bc of `{}`", name);
                    Err(write::llvm_err(&diag_handler, msg))
                }
            })?;
        }
    }

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

    if cgcx.no_landing_pads {
        unsafe {
            llvm::LLVMRustMarkAllFunctionsNounwind(llmod);
        }
    }

    if cgcx.opts.cg.save_temps {
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

        time(cgcx.time_passes, "LTO passes", ||
             llvm::LLVMRunPassManager(pm, llmod));

        llvm::LLVMDisposePassManager(pm);
    }
    debug!("lto done");
    Ok(())
}

fn is_versioned_bytecode_format(bc: &[u8]) -> bool {
    let magic_id_byte_count = link::RLIB_BYTECODE_OBJECT_MAGIC.len();
    return bc.len() > magic_id_byte_count &&
           &bc[..magic_id_byte_count] == link::RLIB_BYTECODE_OBJECT_MAGIC;
}

fn extract_bytecode_format_version(bc: &[u8]) -> u32 {
    let pos = link::RLIB_BYTECODE_OBJECT_VERSION_OFFSET;
    let byte_data = &bc[pos..pos + 4];
    let data = unsafe { read_unaligned(byte_data.as_ptr() as *const u32) };
    u32::from_le(data)
}

fn extract_compressed_bytecode_size_v1(bc: &[u8]) -> u64 {
    let pos = link::RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET;
    let byte_data = &bc[pos..pos + 8];
    let data = unsafe { read_unaligned(byte_data.as_ptr() as *const u64) };
    u64::from_le(data)
}
