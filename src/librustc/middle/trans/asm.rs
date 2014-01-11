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
use middle::trans::expr::*;
use middle::trans::type_of::*;

use middle::trans::type_::Type;

use syntax::ast;

// Take an inline assembly expression and splat it out via LLVM
pub fn trans_inline_asm<'a>(bcx: &'a Block<'a>, ia: &ast::InlineAsm)
                        -> &'a Block<'a> {
    let mut bcx = bcx;
    let mut constraints = ~[];
    let mut cleanups = ~[];
    let mut output_types = ~[];

    // Prepare the output operands
    let outputs = ia.outputs.map(|&(c, out)| {
        constraints.push(c);

        let out_datum = unpack_datum!(bcx, trans_to_datum(bcx, out));
        output_types.push(type_of(bcx.ccx(), out_datum.ty));
        out_datum.val

    });

    for c in cleanups.iter() {
        revoke_clean(bcx, *c);
    }
    cleanups.clear();

    // Now the input operands
    let inputs = ia.inputs.map(|&(c, input)| {
        constraints.push(c);

        unpack_result!(bcx, {
            callee::trans_arg_expr(bcx,
                                   expr_ty(bcx, input),
                                   input,
                                   &mut cleanups,
                                   callee::DontAutorefArg)
        })
    });

    for c in cleanups.iter() {
        revoke_clean(bcx, *c);
    }

    let mut constraints = constraints.connect(",");

    let mut clobbers = getClobbers();
    if !ia.clobbers.is_empty() && !clobbers.is_empty() {
        clobbers = format!("{},{}", ia.clobbers, clobbers);
    } else {
        clobbers.push_str(ia.clobbers);
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

    let r = ia.asm.with_c_str(|a| {
        constraints.with_c_str(|c| {
            InlineAsmCall(bcx, a, c, inputs, output_type, ia.volatile, ia.alignstack, dialect)
        })
    });

    // Again, based on how many outputs we have
    if numOutputs == 1 {
        Store(bcx, r, outputs[0]);
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
