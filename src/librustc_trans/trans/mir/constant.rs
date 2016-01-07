// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use back::abi;
use llvm::ValueRef;
use middle::subst::Substs;
use middle::ty::{Ty, HasTypeFlags};
use rustc::middle::const_eval::ConstVal;
use rustc::mir::repr as mir;
use trans::common::{self, Block, C_bool, C_bytes, C_floating_f64, C_integral, C_str_slice};
use trans::consts;
use trans::expr;
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
        let val = self.trans_constval_inner(bcx, cv, ty, bcx.fcx.param_substs);
        let val = if common::type_is_immediate(ccx, ty) {
            OperandValue::Immediate(val)
        } else if common::type_is_fat_ptr(bcx.tcx(), ty) {
            let data = common::const_get_elt(ccx, val, &[abi::FAT_PTR_ADDR as u32]);
            let extra = common::const_get_elt(ccx, val, &[abi::FAT_PTR_EXTRA as u32]);
            OperandValue::FatPtr(data, extra)
        } else {
            OperandValue::Ref(val)
        };

        assert!(!ty.has_erasable_regions());

        OperandRef {
            ty: ty,
            val: val
        }
    }

    /// Translate ConstVal into a bare LLVM ValueRef.
    fn trans_constval_inner(&mut self,
                            bcx: common::Block<'bcx, 'tcx>,
                            cv: &ConstVal,
                            ty: Ty<'tcx>,
                            param_substs: &'tcx Substs<'tcx>)
                            -> ValueRef
    {
        let ccx = bcx.ccx();
        let llty = type_of::type_of(ccx, ty);
        match *cv {
            ConstVal::Float(v) => C_floating_f64(v, llty),
            ConstVal::Bool(v) => C_bool(ccx, v),
            ConstVal::Int(v) => C_integral(llty, v as u64, true),
            ConstVal::Uint(v) => C_integral(llty, v, false),
            ConstVal::Str(ref v) => C_str_slice(ccx, v.clone()),
            ConstVal::ByteStr(ref v) => consts::addr_of(ccx, C_bytes(ccx, v), 1, "byte_str"),
            ConstVal::Struct(id) | ConstVal::Tuple(id) |
            ConstVal::Array(id, _) | ConstVal::Repeat(id, _) => {
                let expr = bcx.tcx().map.expect_expr(id);
                expr::trans(bcx, expr).datum.val
            },
            ConstVal::Function(did) =>
                self.trans_fn_ref(bcx, ty, param_substs, did).immediate()
        }
    }

    pub fn trans_constant(&mut self,
                          bcx: Block<'bcx, 'tcx>,
                          constant: &mir::Constant<'tcx>)
                          -> OperandRef<'tcx>
    {
        let ty = bcx.monomorphize(&constant.ty);
        match constant.literal {
            mir::Literal::Item { def_id, kind, substs } => {
                let substs = bcx.tcx().mk_substs(bcx.monomorphize(&substs));
                self.trans_item_ref(bcx, ty, kind, substs, def_id)
            }
            mir::Literal::Value { ref value } => {
                self.trans_constval(bcx, value, ty)
            }
        }
    }
}
