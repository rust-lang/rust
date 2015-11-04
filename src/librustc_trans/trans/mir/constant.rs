// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::ValueRef;
use rustc::middle::const_eval::ConstVal;
use rustc_mir::repr as mir;
use trans::consts::{self, TrueConst};
use trans::common::{self, Block};
use trans::type_of;

use super::MirContext;

impl<'bcx, 'tcx> MirContext<'bcx, 'tcx> {
    pub fn trans_constant(&mut self,
                          bcx: Block<'bcx, 'tcx>,
                          constant: &mir::Constant<'tcx>)
                          -> ValueRef
    {
        let ccx = bcx.ccx();
        let constant_ty = bcx.monomorphize(&constant.ty);
        let llty = type_of::type_of(ccx, constant_ty);
        match constant.literal {
            mir::Literal::Item { .. } => {
                unimplemented!()
            }
            mir::Literal::Value { ref value } => {
                match *value {
                    ConstVal::Float(v) => common::C_floating_f64(v, llty),
                    ConstVal::Bool(v) => common::C_bool(ccx, v),
                    ConstVal::Int(v) => common::C_integral(llty, v as u64, true),
                    ConstVal::Uint(v) => common::C_integral(llty, v, false),
                    ConstVal::Str(ref v) => common::C_str_slice(ccx, v.clone()),
                    ConstVal::ByteStr(ref v) => consts::addr_of(ccx,
                                                                common::C_bytes(ccx, v),
                                                                1,
                                                                "byte_str"),
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
                        llval
                    }
                    ConstVal::Function(_) => {
                        unimplemented!()
                    }
                }
            }
        }
    }
}
