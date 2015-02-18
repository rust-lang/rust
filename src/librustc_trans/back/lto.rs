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
use rustc::metadata::cstore;
use rustc::util::common::time;

use libc;
use flate;

use std::ffi::CString;
use std::iter;
use std::mem;
use std::num::Int;

pub fn run(sess: &session::Session, llmod: ModuleRef,
           tm: TargetMachineRef, reachable: &[String]) {
    if sess.opts.cg.prefer_dynamic {
        sess.err("cannot prefer dynamic linking when performing LTO");
        sess.note("only 'staticlib' and 'bin' outputs are supported with LTO");
        sess.abort_if_errors();
    }

    // Make sure we actually can run LTO
    for crate_type in &*sess.crate_types.borrow() {
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
    for (cnum, path) in crates {
        let name = sess.cstore.get_crate_data(cnum).name.clone();
        let path = match path {
            Some(p) => p,
            None => {
                sess.fatal(&format!("could not find rlib for: `{}`",
                                   name)[]);
            }
        };

        let archive = ArchiveRO::open(&path).expect("wanted an rlib");
        let file = path.filename_str().unwrap();
        let file = &file[3..file.len() - 5]; // chop off lib/.rlib
        debug!("reading {}", file);
        for i in iter::count(0, 1) {
            let bc_encoded = time(sess.time_passes(),
                                  &format!("check for {}.{}.bytecode.deflate", name, i),
                                  (),
                                  |_| {
                                      archive.read(&format!("{}.{}.bytecode.deflate",
                                                           file, i)[])
                                  });
            let bc_encoded = match bc_encoded {
                Some(data) => data,
                None => {
                    if i == 0 {
                        // No bitcode was found at all.
                        sess.fatal(&format!("missing compressed bytecode in {}",
                                           path.display())[]);
                    }
                    // No more bitcode files to read.
                    break;
                },
            };

            let bc_decoded = if is_versioned_bytecode_format(bc_encoded) {
                time(sess.time_passes(), &format!("decode {}.{}.bc", file, i), (), |_| {
                    // Read the version
                    let version = extract_bytecode_format_version(bc_encoded);

                    if version == 1 {
                        // The only version existing so far
                        let data_size = extract_compressed_bytecode_size_v1(bc_encoded);
                        let compressed_data = &bc_encoded[
                            link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET..
                            (link::RLIB_BYTECODE_OBJECT_V1_DATA_OFFSET + data_size as uint)];

                        match flate::inflate_bytes(compressed_data) {
                            Some(inflated) => inflated,
                            None => {
                                sess.fatal(&format!("failed to decompress bc of `{}`",
                                                   name)[])
                            }
                        }
                    } else {
                        sess.fatal(&format!("Unsupported bytecode format version {}",
                                           version)[])
                    }
                })
            } else {
                time(sess.time_passes(), &format!("decode {}.{}.bc", file, i), (), |_| {
                // the object must be in the old, pre-versioning format, so simply
                // inflate everything and let LLVM decide if it can make sense of it
                    match flate::inflate_bytes(bc_encoded) {
                        Some(bc) => bc,
                        None => {
                            sess.fatal(&format!("failed to decompress bc of `{}`",
                                               name)[])
                        }
                    }
                })
            };

            let ptr = bc_decoded.as_ptr();
            debug!("linking {}, part {}", name, i);
            time(sess.time_passes(),
                 &format!("ll link {}.{}", name, i)[],
                 (),
                 |()| unsafe {
                if !llvm::LLVMRustLinkInExternalBitcode(llmod,
                                                        ptr as *const libc::c_char,
                                                        bc_decoded.len() as libc::size_t) {
                    write::llvm_err(sess.diagnostic().handler(),
                                    format!("failed to load bc of `{}`",
                                            &name[..]));
                }
            });
        }
    }

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

        let opt = sess.opts.cg.opt_level.unwrap_or(0) as libc::c_uint;

        let builder = llvm::LLVMPassManagerBuilderCreate();
        llvm::LLVMPassManagerBuilderSetOptLevel(builder, opt);
        llvm::LLVMPassManagerBuilderPopulateLTOPassManager(builder, pm,
            /* Internalize = */ False,
            /* RunInliner = */ True);
        llvm::LLVMPassManagerBuilderDispose(builder);

        llvm::LLVMRustAddPass(pm, "verify\0".as_ptr() as *const _);

        time(sess.time_passes(), "LTO passes", (), |()|
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
    return read_from_le_bytes::<u32>(bc, link::RLIB_BYTECODE_OBJECT_VERSION_OFFSET);
}

fn extract_compressed_bytecode_size_v1(bc: &[u8]) -> u64 {
    return read_from_le_bytes::<u64>(bc, link::RLIB_BYTECODE_OBJECT_V1_DATASIZE_OFFSET);
}

fn read_from_le_bytes<T: Int>(bytes: &[u8], position_in_bytes: uint) -> T {
    let byte_data = &bytes[position_in_bytes..position_in_bytes + mem::size_of::<T>()];
    let data = unsafe {
        *(byte_data.as_ptr() as *const T)
    };

    Int::from_le(data)
}

