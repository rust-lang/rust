// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ffi::CString;
use std::ptr;

use attributes;
use libc::c_uint;
use rustc::middle::allocator::AllocatorKind;
use rustc::ty::TyCtxt;
use rustc_allocator::{ALLOCATOR_METHODS, AllocatorTy};

use ModuleLlvm;
use llvm::{self, False, True};

pub(crate) unsafe fn codegen(tcx: TyCtxt, mods: &ModuleLlvm, kind: AllocatorKind) {
    let llcx = mods.llcx;
    let llmod = mods.llmod;
    let usize = match &tcx.sess.target.target.target_pointer_width[..] {
        "16" => llvm::LLVMInt16TypeInContext(llcx),
        "32" => llvm::LLVMInt32TypeInContext(llcx),
        "64" => llvm::LLVMInt64TypeInContext(llcx),
        tws => bug!("Unsupported target word size for int: {}", tws),
    };
    let i8 = llvm::LLVMInt8TypeInContext(llcx);
    let i8p = llvm::LLVMPointerType(i8, 0);
    let void = llvm::LLVMVoidTypeInContext(llcx);

    for method in ALLOCATOR_METHODS {
        let mut args = Vec::new();
        for ty in method.inputs.iter() {
            match *ty {
                AllocatorTy::Layout => {
                    args.push(usize); // size
                    args.push(usize); // align
                }
                AllocatorTy::Ptr => args.push(i8p),
                AllocatorTy::Usize => args.push(usize),

                AllocatorTy::ResultPtr |
                AllocatorTy::Unit => panic!("invalid allocator arg"),
            }
        }
        let output = match method.output {
            AllocatorTy::ResultPtr => Some(i8p),
            AllocatorTy::Unit => None,

            AllocatorTy::Layout |
            AllocatorTy::Usize |
            AllocatorTy::Ptr => panic!("invalid allocator output"),
        };
        let ty = llvm::LLVMFunctionType(output.unwrap_or(void),
                                        args.as_ptr(),
                                        args.len() as c_uint,
                                        False);
        let name = CString::new(format!("__rust_{}", method.name)).unwrap();
        let llfn = llvm::LLVMRustGetOrInsertFunction(llmod,
                                                     name.as_ptr(),
                                                     ty);

        if tcx.sess.target.target.options.default_hidden_visibility {
            llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
        }
       if tcx.sess.target.target.options.requires_uwtable {
           attributes::emit_uwtable(llfn, true);
       }

        let callee = CString::new(kind.fn_name(method.name)).unwrap();
        let callee = llvm::LLVMRustGetOrInsertFunction(llmod,
                                                       callee.as_ptr(),
                                                       ty);

        let llbb = llvm::LLVMAppendBasicBlockInContext(llcx,
                                                       llfn,
                                                       "entry\0".as_ptr() as *const _);

        let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
        llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
        let args = args.iter().enumerate().map(|(i, _)| {
            llvm::LLVMGetParam(llfn, i as c_uint)
        }).collect::<Vec<_>>();
        let ret = llvm::LLVMRustBuildCall(llbuilder,
                                          callee,
                                          args.as_ptr(),
                                          args.len() as c_uint,
                                          ptr::null_mut(),
                                          "\0".as_ptr() as *const _);
        llvm::LLVMSetTailCall(ret, True);
        if output.is_some() {
            llvm::LLVMBuildRet(llbuilder, ret);
        } else {
            llvm::LLVMBuildRetVoid(llbuilder);
        }
        llvm::LLVMDisposeBuilder(llbuilder);
    }
}
