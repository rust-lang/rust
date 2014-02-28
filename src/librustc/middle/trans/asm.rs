// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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

use std::c_str::ToCStr;

use lib;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::trans::cleanup;
use middle::trans::cleanup::CleanupMethods;
use middle::trans::expr;
use middle::trans::type_of;

use middle::trans::type_::Type;

use syntax::ast;

// Take an inline assembly expression and splat it out via LLVM
pub fn trans_inline_asm<'a>(bcx: &'a Block<'a>, ia: &ast::InlineAsm)
                        -> &'a Block<'a> {
    let fcx = bcx.fcx;
    let mut bcx = bcx;
    let mut constraints = ~[];
    let mut output_types = ~[];

    let temp_scope = fcx.push_custom_cleanup_scope();

    // Prepare the output operands
    let outputs = ia.outputs.map(|&(ref c, out)| {
        constraints.push((*c).clone());

        let out_datum = unpack_datum!(bcx, expr::trans(bcx, out));
        output_types.push(type_of::type_of(bcx.ccx(), out_datum.ty));
        out_datum.val

    });

    // Now the input operands
    let inputs = ia.inputs.map(|&(ref c, input)| {
        constraints.push((*c).clone());

        unpack_result!(bcx, {
            callee::trans_arg_expr(bcx,
                                   expr_ty(bcx, input),
                                   input,
                                   cleanup::CustomScope(temp_scope),
                                   callee::DontAutorefArg)
        })
    });

    // no failure occurred preparing operands, no need to cleanup
    fcx.pop_custom_cleanup_scope(temp_scope);

    let mut constraints = constraints.map(|s| s.get().to_str()).connect(",");

    let mut clobbers = getClobbers();
    if !ia.clobbers.get().is_empty() && !clobbers.is_empty() {
        clobbers = format!("{},{}", ia.clobbers.get(), clobbers);
    } else {
        clobbers.push_str(ia.clobbers.get());
    }

    // Add the clobbers to our constraints list
    if clobbers.len() != 0 && constraints.len() != 0 {
        constraints.push_char(',');
        constraints.push_str(clobbers);
    } else {
        constraints.push_str(clobbers);
    }

    debug!("Asm Constraints: {:?}", constraints);

    let numOutputs = outputs.len();

    // Depending on how many outputs we have, the return type is different
    let output_type = if numOutputs == 0 {
        Type::void()
    } else if numOutputs == 1 {
        output_types[0]
    } else {
        Type::struct_(output_types, false)
    };

    let dialect = match ia.dialect {
        ast::AsmAtt   => lib::llvm::AD_ATT,
        ast::AsmIntel => lib::llvm::AD_Intel
    };

    let r = ia.asm.get().with_c_str(|a| {
        constraints.with_c_str(|c| {
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
    if numOutputs == 1 {
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
fn getClobbers() -> ~str {
    ~""
}

#[cfg(target_arch = "x86")]
#[cfg(target_arch = "x86_64")]
fn getClobbers() -> ~str {
    ~"~{dirflag},~{fpsr},~{flags}"
}
