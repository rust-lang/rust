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

use core::prelude::*;

use lib;
use middle::trans::build::*;
use middle::trans::callee;
use middle::trans::common::*;
use middle::ty;

use core::str;
use syntax::ast;

// Take an inline assembly expression and splat it out via LLVM
pub fn trans_inline_asm(bcx: block, ia: &ast::inline_asm) -> block {

    let mut bcx = bcx;
    let mut constraints = ~[];
    let mut cleanups = ~[];
    let mut aoutputs = ~[];

    // Prepare the output operands
    let outputs = do ia.outputs.map |&(c, out)| {
        constraints.push(copy *c);

        aoutputs.push(unpack_result!(bcx, {
            callee::trans_arg_expr(bcx,
                                   expr_ty(bcx, out),
                                   ty::ByCopy,
                                   out,
                                   &mut cleanups,
                                   None,
                                   callee::DontAutorefArg)
        }));

        let e = match out.node {
            ast::expr_addr_of(_, e) => e,
            _ => fail!("Expression must be addr of")
        };

        unpack_result!(bcx, {
            callee::trans_arg_expr(bcx,
                                   expr_ty(bcx, e),
                                   ty::ByCopy,
                                   e,
                                   &mut cleanups,
                                   None,
                                   callee::DontAutorefArg)
        })

    };

    for cleanups.each |c| {
        revoke_clean(bcx, *c);
    }
    cleanups.clear();

    // Now the input operands
    let inputs = do ia.inputs.map |&(c, in)| {
        constraints.push(copy *c);

        unpack_result!(bcx, {
            callee::trans_arg_expr(bcx,
                                   expr_ty(bcx, in),
                                   ty::ByCopy,
                                   in,
                                   &mut cleanups,
                                   None,
                                   callee::DontAutorefArg)
        })

    };

    for cleanups.each |c| {
        revoke_clean(bcx, *c);
    }

    let mut constraints = constraints.connect(",");

    let mut clobbers = getClobbers();
    if *ia.clobbers != ~"" && clobbers != ~"" {
        clobbers = *ia.clobbers + "," + clobbers;
    } else {
        clobbers += *ia.clobbers;
    };

    // Add the clobbers to our constraints list
    if clobbers != ~"" && constraints != ~"" {
        constraints += ",";
        constraints += clobbers;
    } else {
        constraints += clobbers;
    }

    debug!("Asm Constraints: %?", constraints);

    let numOutputs = outputs.len();

    // Depending on how many outputs we have, the return type is different
    let output = if numOutputs == 0 {
        T_void()
    } else if numOutputs == 1 {
        val_ty(outputs[0])
    } else {
        T_struct(outputs.map(|o| val_ty(*o)), false)
    };

    let dialect = match ia.dialect {
        ast::asm_att   => lib::llvm::AD_ATT,
        ast::asm_intel => lib::llvm::AD_Intel
    };

    let r = do str::as_c_str(*ia.asm) |a| {
        do str::as_c_str(constraints) |c| {
            InlineAsmCall(bcx, a, c, inputs, output, ia.volatile, ia.alignstack, dialect)
        }
    };

    // Again, based on how many outputs we have
    if numOutputs == 1 {
        let op = PointerCast(bcx, aoutputs[0], T_ptr(val_ty(outputs[0])));
        Store(bcx, r, op);
    } else {
        for aoutputs.eachi |i, o| {
            let v = ExtractValue(bcx, r, i);
            let op = PointerCast(bcx, *o, T_ptr(val_ty(outputs[i])));
            Store(bcx, v, op);
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
