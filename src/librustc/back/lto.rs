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
use driver::session;
use driver::config;
use llvm;
use llvm::archive_ro::ArchiveRO;
use llvm::{ModuleRef, TargetMachineRef, True, False};
use metadata::cstore;
use util::common::time;

use libc;
use flate;

use std::mem;

pub fn run(sess: &session::Session, llmod: ModuleRef,
           tm: TargetMachineRef, reachable: &[String]) {
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
    let crates = sess.cstore.get_used_crates(cstore::RequireStatic);
    for (cnum, path) in crates.into_iter() {
        let name = sess.cstore.get_crate_data(cnum).name.clone();
        let path = match path {
            Some(p) => p,
            None => {
                sess.fatal(format!("could not find rlib for: `{}`",
                                   name).as_slice());
            }
        };

        let archive = ArchiveRO::open(&path).expect("wanted an rlib");
        let file = path.filename_str().unwrap();
        let file = file.slice(3, file.len() - 5); // chop off lib/.rlib
        debug!("reading {}", file);
        let bc_encoded = time(sess.time_passes(),
                              format!("read {}.bytecode.deflate", name).as_slice(),
                              (),
                              |_| {
                                  archive.read(format!("{}.bytecode.deflate",
                                                       file).as_slice())
                              });
        let bc_encoded = match bc_encoded {
            Some(data) => data,
            None => {
                sess.fatal(format!("missing compressed bytecode in {} \
                                    (perhaps it was compiled with -C codegen-units > 1)",
                                   path.display()).as_slice());
            },
        };
        let bc_extractor = if is_versioned_bytecode_format(bc_encoded) {
            |_| {
                // Read the version
                let version = extract_bytecode_format_version(bc_encoded);

                if version == 1 {
                    // The only version existing so far
                    let data_size = extract_compressed_bytecode_size_v1(bc_encoded);
                    let compressed_data = bc_encoded.slice(
                        link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET,
                        link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET + data_size as uint);

                    match flate::inflate_bytes(compressed_data) {
                        Some(inflated) => inflated,
                        None => {
                            sess.fatal(format!("failed to decompress bc of `{}`",
                                               name).as_slice())
                        }
                    }
                } else {
                    sess.fatal(format!("Unsupported bytecode format version {}",
                                       version).as_slice())
                }
            }
        } else {
            // the object must be in the old, pre-versioning format, so simply
            // inflate everything and let LLVM decide if it can make sense of it
            |_| {
                match flate::inflate_bytes(bc_encoded) {
                    Some(bc) => bc,
                    None => {
                        sess.fatal(format!("failed to decompress bc of `{}`",
                                           name).as_slice())
                    }
                }
            }
        };

        let bc_decoded = time(sess.time_passes(),
                              format!("decode {}.bc", file).as_slice(),
                              (),
                              bc_extractor);

        let ptr = bc_decoded.as_slice().as_ptr();
        debug!("linking {}", name);
        time(sess.time_passes(),
             format!("ll link {}", name).as_slice(),
             (),
             |()| unsafe {
            if !llvm::LLVMRustLinkInExternalBitcode(llmod,
                                                    ptr as *const libc::c_char,
                                                    bc_decoded.len() as libc::size_t) {
                write::llvm_err(sess.diagnostic().handler(),
                                format!("failed to load bc of `{}`",
                                        name.as_slice()));
            }
        });
    }

    // Internalize everything but the reachable symbols of the current module
    let cstrs: Vec<::std::c_str::CString> =
        reachable.iter().map(|s| s.as_slice().to_c_str()).collect();
    let arr: Vec<*const i8> = cstrs.iter().map(|c| c.as_ptr()).collect();
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

    // Now we have one massive module inside of llmod. Time to run the
    // LTO-specific optimization passes that LLVM provides.
    //
    // This code is based off the code found in llvm's LTO code generator:
    //      tools/lto/LTOCodeGenerator.cpp
    debug!("running the pass manager");
    unsafe {
        let pm = llvm::LLVMCreatePassManager();
        llvm::LLVMRustAddAnalysisPasses(tm, pm, llmod);
        "verify".with_c_str(|s| llvm::LLVMRustAddPass(pm, s));

        let builder = llvm::LLVMPassManagerBuilderCreate();
        llvm::LLVMPassManagerBuilderPopulateLTOPassManager(builder, pm,
            /* Internalize = */ False,
            /* RunInliner = */ True);
        llvm::LLVMPassManagerBuilderDispose(builder);

        "verify".with_c_str(|s| llvm::LLVMRustAddPass(pm, s));

        time(sess.time_passes(), "LTO passes", (), |()|
             llvm::LLVMRunPassManager(pm, llmod));

        llvm::LLVMDisposePassManager(pm);
    }
    debug!("lto done");
}

fn is_versioned_bytecode_format(bc: &[u8]) -> bool {
    let magic_id_byte_count = link::RLIB_BYTECODE_OBJECT_MAGIC.len();
    return bc.len() > magic_id_byte_count &&
           bc.slice(0, magic_id_byte_count) == link::RLIB_BYTECODE_OBJECT_MAGIC;
}

fn extract_bytecode_format_version(bc: &[u8]) -> u32 {
    return read_from_le_bytes::<u32>(bc, link::RLIB_BYTECODE_OBJECT_VERSION_OFFSET);
}

fn extract_compressed_bytecode_size_v1(bc: &[u8]) -> u64 {
    return read_from_le_bytes::<u64>(bc, link::RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET);
}

fn read_from_le_bytes<T: Int>(bytes: &[u8], position_in_bytes: uint) -> T {
    let byte_data = bytes.slice(position_in_bytes,
                                position_in_bytes + mem::size_of::<T>());
    let data = unsafe {
        *(byte_data.as_ptr() as *const T)
    };

    Int::from_le(data)
}

