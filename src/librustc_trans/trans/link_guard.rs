// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::svh::Svh;
use libc::c_uint;
use llvm;
use std::ffi::CString;
use std::ptr;
use trans::attributes;
use trans::builder;
use trans::CrateContext;
use trans::declare;
use trans::type_::Type;

const GUARD_PREFIX: &'static str = "__rustc_link_guard_";

pub fn link_guard_name(crate_name: &str, crate_svh: &Svh) -> String {

    let mut guard_name = String::new();

    guard_name.push_str(GUARD_PREFIX);
    guard_name.push_str(crate_name);
    guard_name.push_str("_");
    guard_name.push_str(crate_svh.as_str());

    guard_name
}

pub fn get_or_insert_link_guard<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>)
                                          -> llvm::ValueRef {

    let guard_name = link_guard_name(&ccx.tcx().crate_name[..],
                                     &ccx.link_meta().crate_hash);

    let guard_function = unsafe {
        let guard_name_c_string = CString::new(&guard_name[..]).unwrap();
        llvm::LLVMGetNamedValue(ccx.llmod(), guard_name_c_string.as_ptr())
    };

    if guard_function != ptr::null_mut() {
        return guard_function;
    }

    let llfty = Type::func(&[], &Type::void(ccx));
    let guard_function = declare::define_cfn(ccx,
                                             &guard_name[..],
                                             llfty,
                                             ccx.tcx().mk_nil()).unwrap_or_else(|| {
        ccx.sess().bug("Link guard already defined.");
    });

    attributes::emit_uwtable(guard_function, true);
    attributes::unwind(guard_function, false);

    let bld = ccx.raw_builder();
    unsafe {
        let llbb = llvm::LLVMAppendBasicBlockInContext(ccx.llcx(),
                                                       guard_function,
                                                       "link_guard_top\0".as_ptr() as *const _);
        llvm::LLVMPositionBuilderAtEnd(bld, llbb);

        for crate_num in ccx.sess().cstore.crates() {
            if !ccx.sess().cstore.is_explicitly_linked(crate_num) {
                continue;
            }

            let crate_name = ccx.sess().cstore.original_crate_name(crate_num);
            let svh = ccx.sess().cstore.crate_hash(crate_num);

            let dependency_guard_name = link_guard_name(&crate_name[..], &svh);

            let decl = declare::declare_cfn(ccx,
                                            &dependency_guard_name[..],
                                            llfty,
                                            ccx.tcx().mk_nil());
            attributes::unwind(decl, false);

            llvm::LLVMPositionBuilderAtEnd(bld, llbb);

            let args: &[llvm::ValueRef] = &[];
            llvm::LLVMRustBuildCall(bld,
                                    decl,
                                    args.as_ptr(),
                                    args.len() as c_uint,
                                    0 as *mut _,
                                    builder::noname());
        }

        llvm::LLVMBuildRetVoid(bld);
    }

    guard_function
}

pub fn insert_reference_to_link_guard<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                                llbb: llvm::BasicBlockRef) {
    let guard_function = get_or_insert_link_guard(ccx);

    unsafe {
        llvm::LLVMPositionBuilderAtEnd(ccx.raw_builder(), llbb);
        let args: &[llvm::ValueRef] = &[];
        llvm::LLVMRustBuildCall(ccx.raw_builder(),
                                guard_function,
                                args.as_ptr(),
                                args.len() as c_uint,
                                0 as *mut _,
                                builder::noname());
    }
}
