// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm;
use context::CodegenCx;
use type_of::LayoutLlvmExt;
use builder::Builder;
use value::Value;

use rustc::hir;
use interfaces::{BuilderMethods, CommonMethods, TypeMethods};

use mir::place::PlaceRef;
use mir::operand::OperandValue;

use std::ffi::CString;
use libc::{c_uint, c_char};

// Take an inline assembly expression and splat it out via LLVM
pub fn codegen_inline_asm(
    bx: &Builder<'a, 'll, 'tcx, &'ll Value>,
    ia: &hir::InlineAsm,
    outputs: Vec<PlaceRef<'tcx, &'ll Value>>,
    mut inputs: Vec<&'ll Value>
) -> bool {
    let mut ext_constraints = vec![];
    let mut output_types = vec![];

    // Prepare the output operands
    let mut indirect_outputs = vec![];
    for (i, (out, place)) in ia.outputs.iter().zip(&outputs).enumerate() {
        if out.is_rw {
            inputs.push(place.load(bx).immediate());
            ext_constraints.push(i.to_string());
        }
        if out.is_indirect {
            indirect_outputs.push(place.load(bx).immediate());
        } else {
            output_types.push(place.layout.llvm_type(bx.cx()));
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
    let arch_clobbers = match &bx.sess().target.target.arch[..] {
        "x86" | "x86_64"  => vec!["~{dirflag}", "~{fpsr}", "~{flags}"],
        "mips" | "mips64" => vec!["~{$1}"],
        _                 => Vec::new()
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
        0 => bx.cx().void(),
        1 => output_types[0],
        _ => bx.cx().struct_(&output_types, false)
    };

    let asm = CString::new(ia.asm.as_str().as_bytes()).unwrap();
    let constraint_cstr = CString::new(all_constraints).unwrap();
    let r = bx.inline_asm_call(
        asm.as_ptr(),
        constraint_cstr.as_ptr(),
        &inputs,
        output_type,
        ia.volatile,
        ia.alignstack,
        ia.dialect
    );
    if r.is_none() {
        return false;
    }
    let r = r.unwrap();

    // Again, based on how many outputs we have
    let outputs = ia.outputs.iter().zip(&outputs).filter(|&(ref o, _)| !o.is_indirect);
    for (i, (_, &place)) in outputs.enumerate() {
        let v = if num_outputs == 1 { r } else { bx.extract_value(r, i as u64) };
        OperandValue::Immediate(v).store(bx, place);
    }

    // Store mark in a metadata node so we can map LLVM errors
    // back to source locations.  See #17552.
    unsafe {
        let key = "srcloc";
        let kind = llvm::LLVMGetMDKindIDInContext(bx.cx().llcx,
            key.as_ptr() as *const c_char, key.len() as c_uint);

        let val: &'ll Value = bx.cx().c_i32(ia.ctxt.outer().as_u32() as i32);

        llvm::LLVMSetMetadata(r, kind,
            llvm::LLVMMDNodeInContext(bx.cx().llcx, &val, 1));
    }

    return true;
}

pub fn codegen_global_asm<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx, &'a Value>,
                                  ga: &hir::GlobalAsm) {
    let asm = CString::new(ga.asm.as_str().as_bytes()).unwrap();
    unsafe {
        llvm::LLVMRustAppendModuleInlineAsm(cx.llmod, asm.as_ptr());
    }
}
