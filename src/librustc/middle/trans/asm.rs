// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
# Translation of inline assembly.
*/

use lib;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::expr;
use middle::trans::type_of;
use middle::trans::type_::Type;

use std::c_str::ToCStr;
use std::string::String;
use syntax::ast;

// Take an inline assembly expression and splat it out via LLVM
pub fn trans_inline_asm<'a>(bcx: &'a Block<'a>, ia: &ast::InlineAsm)
                        -> &'a Block<'a> {
    let fcx = bcx.fcx;
    let mut bcx = bcx;
    let mut constraints = Vec::new();
    let mut output_types = Vec::new();

    let temp_scope = fcx.push_custom_cleanup_scope();

    // Prepare the output operands
    let outputs = ia.outputs.iter().map(|&(ref c, ref out)| {
        constraints.push((*c).clone());

        let out_datum = unpack_datum!(bcx, expr::trans(bcx, &**out));
        output_types.push(type_of::type_of(bcx.ccx(), out_datum.ty));
        out_datum.val

    }).collect::<Vec<_>>();

    // Now the input operands
    let inputs = ia.inputs.iter().map(|&(ref c, ref input)| {
        constraints.push((*c).clone());

        let in_datum = unpack_datum!(bcx, expr::trans(bcx, &**input));
        unpack_result!(bcx, {
            callee::trans_arg_datum(bcx,
                                   expr_ty(bcx, &**input),
                                   in_datum,
                                   cleanup::CustomScope(temp_scope),
                                   callee::DontAutorefArg)
        })
    }).collect::<Vec<_>>();

    // no failure occurred preparing operands, no need to cleanup
    fcx.pop_custom_cleanup_scope(temp_scope);

    let mut constraints =
        String::from_str(constraints.iter()
                                    .map(|s| s.get().to_string())
                                    .collect::<Vec<String>>()
                                    .connect(",")
                                    .as_slice());

    let mut clobbers = get_clobbers();
    if !ia.clobbers.get().is_empty() && !clobbers.is_empty() {
        clobbers = format!("{},{}", ia.clobbers.get(), clobbers);
    } else {
        clobbers.push_str(ia.clobbers.get());
    }

    // Add the clobbers to our constraints list
    if clobbers.len() != 0 && constraints.len() != 0 {
        constraints.push_char(',');
        constraints.push_str(clobbers.as_slice());
    } else {
        constraints.push_str(clobbers.as_slice());
    }

    debug!("Asm Constraints: {:?}", constraints.as_slice());

    let num_outputs = outputs.len();

    // Depending on how many outputs we have, the return type is different
    let output_type = if num_outputs == 0 {
        Type::void(bcx.ccx())
    } else if num_outputs == 1 {
        *output_types.get(0)
    } else {
        Type::struct_(bcx.ccx(), output_types.as_slice(), false)
    };

    let dialect = match ia.dialect {
        ast::AsmAtt   => lib::llvm::AD_ATT,
        ast::AsmIntel => lib::llvm::AD_Intel
    };

    let r = ia.asm.get().with_c_str(|a| {
        constraints.as_slice().with_c_str(|c| {
            InlineAsmCall(bcx,
                          a,
                          c,
                          inputs.as_slice(),
                          output_type,
                          ia.volatile,
                          ia.alignstack,
                          dialect)
        })
    });

    // Again, based on how many outputs we have
    if num_outputs == 1 {
        Store(bcx, r, *outputs.get(0));
    } else {
        for (i, o) in outputs.iter().enumerate() {
            let v = ExtractValue(bcx, r, i);
            Store(bcx, v, *o);
        }
    }

    return bcx;

}

// Default per-arch clobbers
// Basically what clang does

#[cfg(target_arch = "arm")]
#[cfg(target_arch = "mips")]
#[cfg(target_arch = "mipsel")]
fn get_clobbers() -> String {
    "".to_string()
}

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "x86_64")]
fn get_clobbers() -> String {
    "~{dirflag},~{fpsr},~{flags}".to_string()
}
