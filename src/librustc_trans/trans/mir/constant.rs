// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::{Ty, HasTypeFlags};
use rustc::middle::const_eval::ConstVal;
use rustc::mir::repr as mir;
use trans::consts::{self, TrueConst};
use trans::common::{self, Block};
use trans::common::{C_bool, C_bytes, C_floating_f64, C_integral, C_str_slice};
use trans::type_of;

use super::operand::OperandRef;
use super::MirContext;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_constval(&mut self,
                          bcx: Block<'bcx, 'tcx>,
                          cv: &ConstVal,
                          ty: Ty<'tcx>)
                          -> OperandRef<'tcx>
    {
        use super::operand::OperandValue::{Ref, Immediate};

        let ccx = bcx.ccx();
        let llty = type_of::type_of(ccx, ty);
        let val = match *cv {
            ConstVal::Float(v) => Immediate(C_floating_f64(v, llty)),
            ConstVal::Bool(v) => Immediate(C_bool(ccx, v)),
            ConstVal::Int(v) => Immediate(C_integral(llty, v as u64, true)),
            ConstVal::Uint(v) => Immediate(C_integral(llty, v, false)),
            ConstVal::Str(ref v) => Immediate(C_str_slice(ccx, v.clone())),
            ConstVal::ByteStr(ref v) => {
                Immediate(consts::addr_of(ccx,
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
                    Immediate(llval)
                } else {
                    Ref(llval)
                }
            }
            ConstVal::Function(_) => {
                unimplemented!()
            }
            ConstVal::Array(..) => {
                unimplemented!()
            }
            ConstVal::Repeat(..) => {
                unimplemented!()
            }
        };

        assert!(!ty.has_erasable_regions());

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
