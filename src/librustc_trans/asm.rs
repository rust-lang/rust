// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Translation of inline assembly.

use llvm::{self, ValueRef};
use common::*;
use type_::Type;
use builder::Builder;

use rustc::hir;

use mir::lvalue::LvalueRef;
use mir::operand::OperandValue;

use std::ffi::CString;
use syntax::ast::AsmDialect;
use libc::{c_uint, c_char};

// Take an inline assembly expression and splat it out via LLVM
pub fn trans_inline_asm<'a, 'tcx>(
    bcx: &Builder<'a, 'tcx>,
    ia: &hir::InlineAsm,
    outputs: Vec<LvalueRef<'tcx>>,
    mut inputs: Vec<ValueRef>
) {
    let mut ext_constraints = vec![];
    let mut output_types = vec![];

    // Prepare the output operands
    let mut indirect_outputs = vec![];
    for (i, (out, lvalue)) in ia.outputs.iter().zip(&outputs).enumerate() {
        if out.is_rw {
            inputs.push(lvalue.load(bcx).immediate());
            ext_constraints.push(i.to_string());
        }
        if out.is_indirect {
            indirect_outputs.push(lvalue.load(bcx).immediate());
        } else {
            output_types.push(bcx.ccx.llvm_type_of(lvalue.layout.ty));
        }
    }
    if !indirect_outputs.is_empty() {
        indirect_outputs.extend_from_slice(&inputs);
        inputs = indirect_outputs;
    }

    let clobbers = ia.clobbers.iter()
                              .map(|s| format!("~{{{}}}", &s));

    // Default per-arch clobbers
    // Basically what clang does
    let arch_clobbers = match &bcx.sess().target.target.arch[..] {
        "x86" | "x86_64" => vec!["~{dirflag}", "~{fpsr}", "~{flags}"],
        _                => Vec::new()
    };

    let all_constraints =
        ia.outputs.iter().map(|out| out.constraint.to_string())
          .chain(ia.inputs.iter().map(|s| s.to_string()))
          .chain(ext_constraints)
          .chain(clobbers)
          .chain(arch_clobbers.iter().map(|s| s.to_string()))
          .collect::<Vec<String>>().join(",");

    debug!("Asm Constraints: {}", &all_constraints);

    // Depending on how many outputs we have, the return type is different
    let num_outputs = output_types.len();
    let output_type = match num_outputs {
        0 => Type::void(bcx.ccx),
        1 => output_types[0],
        _ => Type::struct_(bcx.ccx, &output_types, false)
    };

    let dialect = match ia.dialect {
        AsmDialect::Att   => llvm::AsmDialect::Att,
        AsmDialect::Intel => llvm::AsmDialect::Intel,
    };

    let asm = CString::new(ia.asm.as_str().as_bytes()).unwrap();
    let constraint_cstr = CString::new(all_constraints).unwrap();
    let r = bcx.inline_asm_call(
        asm.as_ptr(),
        constraint_cstr.as_ptr(),
        &inputs,
        output_type,
        ia.volatile,
        ia.alignstack,
        dialect
    );

    // Again, based on how many outputs we have
    let outputs = ia.outputs.iter().zip(&outputs).filter(|&(ref o, _)| !o.is_indirect);
    for (i, (_, &lvalue)) in outputs.enumerate() {
        let v = if num_outputs == 1 { r } else { bcx.extract_value(r, i as u64) };
        OperandValue::Immediate(v).store(bcx, lvalue);
    }

    // Store mark in a metadata node so we can map LLVM errors
    // back to source locations.  See #17552.
    unsafe {
        let key = "srcloc";
        let kind = llvm::LLVMGetMDKindIDInContext(bcx.ccx.llcx(),
            key.as_ptr() as *const c_char, key.len() as c_uint);

        let val: llvm::ValueRef = C_i32(bcx.ccx, ia.ctxt.outer().as_u32() as i32);

        llvm::LLVMSetMetadata(r, kind,
            llvm::LLVMMDNodeInContext(bcx.ccx.llcx(), &val, 1));
    }
}

pub fn trans_global_asm<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                  ga: &hir::GlobalAsm) {
    let asm = CString::new(ga.asm.as_str().as_bytes()).unwrap();
    unsafe {
        llvm::LLVMRustAppendModuleInlineAsm(ccx.llmod(), asm.as_ptr());
    }
}
