// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::Ty;
use rustc::middle::const_eval::ConstVal;
use rustc_mir::repr as mir;
use trans::consts::{self, TrueConst};
use trans::common::{self, Block};
use trans::common::{C_bool, C_bytes, C_floating_f64, C_integral, C_str_slice};
use trans::type_of;

use super::operand::{OperandRef, OperandValue};
use super::MirContext;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_constval(&mut self,
                          bcx: Block<'bcx, 'tcx>,
                          cv: &ConstVal,
                          ty: Ty<'tcx>)
                          -> OperandRef<'tcx>
    {
        let ccx = bcx.ccx();
        let llty = type_of::type_of(ccx, ty);
        let val = match *cv {
            ConstVal::Float(v) => OperandValue::Imm(C_floating_f64(v, llty)),
            ConstVal::Bool(v) => OperandValue::Imm(C_bool(ccx, v)),
            ConstVal::Int(v) => OperandValue::Imm(C_integral(llty, v as u64, true)),
            ConstVal::Uint(v) => OperandValue::Imm(C_integral(llty, v, false)),
            ConstVal::Str(ref v) => OperandValue::Imm(C_str_slice(ccx, v.clone())),
            ConstVal::ByteStr(ref v) => {
                OperandValue::Imm(consts::addr_of(ccx,
                                                  C_bytes(ccx, v),
                                                  1,
                                                  "byte_str"))
            }

            ConstVal::Struct(id) | ConstVal::Tuple(id) => {
                let expr = bcx.tcx().map.expect_expr(id);
                let (llval, _) = match consts::const_expr(ccx,
                                                          expr,
                                                          bcx.fcx.param_substs,
                                                          None,
                                                          TrueConst::Yes) {
                    Ok(v) => v,
                    Err(_) => panic!("constant eval failure"),
                };
                if common::type_is_immediate(bcx.ccx(), ty) {
                    OperandValue::Imm(llval)
                } else {
                    OperandValue::Ref(llval)
                }
            }
            ConstVal::Function(_) => {
                unimplemented!()
            }
        };
        OperandRef {
            ty: ty,
            val: val
        }
    }

    pub fn trans_constant(&mut self,
                          bcx: Block<'bcx, 'tcx>,
                          constant: &mir::Constant<'tcx>)
                          -> OperandRef<'tcx>
    {
        let constant_ty = bcx.monomorphize(&constant.ty);
        match constant.literal {
            mir::Literal::Item { .. } => {
                unimplemented!()
            }
            mir::Literal::Value { ref value } => {
                self.trans_constval(bcx, value, constant_ty)
            }
        }
    }
}
