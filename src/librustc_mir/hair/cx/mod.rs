// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * This module contains the code to convert from the wacky tcx data
 * structures into the hair. The `builder` is generally ignorant of
 * the tcx etc, and instead goes through the `Cx` for most of its
 * work.
 */

use hair::*;
use rustc::mir::repr::*;

use rustc::middle::const_eval::{self, ConstVal};
use rustc::middle::def_id::DefId;
use rustc::middle::infer::InferCtxt;
use rustc::middle::subst::{Subst, Substs};
use rustc::middle::ty::{self, Ty, TyCtxt};
use syntax::codemap::Span;
use syntax::parse::token;
use rustc_front::hir;
use rustc_const_eval::{ConstInt, ConstUsize};

#[derive(Copy, Clone)]
pub struct Cx<'a, 'tcx: 'a> {
    tcx: &'a TyCtxt<'tcx>,
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a,'tcx> Cx<'a,'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Cx<'a, 'tcx> {
        Cx {
            tcx: infcx.tcx,
            infcx: infcx,
        }
    }
}

impl<'a,'tcx:'a> Cx<'a, 'tcx> {
    /// Normalizes `ast` into the appropriate `mirror` type.
    pub fn mirror<M: Mirror<'tcx>>(&mut self, ast: M) -> M::Output {
        ast.make_mirror(self)
    }

    pub fn usize_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.usize
    }

    pub fn usize_literal(&mut self, value: u64) -> Literal<'tcx> {
        match ConstUsize::new(value, self.tcx.sess.target.uint_type) {
            Ok(val) => Literal::Value { value: ConstVal::Integral(ConstInt::Usize(val))},
            Err(_) => panic!("usize literal out of range for target"),
        }
    }

    pub fn bool_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.bool
    }

    pub fn unit_ty(&mut self) -> Ty<'tcx> {
        self.tcx.mk_nil()
    }

    pub fn str_literal(&mut self, value: token::InternedString) -> Literal<'tcx> {
        Literal::Value { value: ConstVal::Str(value) }
    }

    pub fn true_literal(&mut self) -> Literal<'tcx> {
        Literal::Value { value: ConstVal::Bool(true) }
    }

    pub fn false_literal(&mut self) -> Literal<'tcx> {
        Literal::Value { value: ConstVal::Bool(false) }
    }

    pub fn const_eval_literal(&mut self, e: &hir::Expr) -> Literal<'tcx> {
        Literal::Value { value: const_eval::eval_const_expr(self.tcx, e) }
    }

    pub fn try_const_eval_literal(&mut self, e: &hir::Expr) -> Option<Literal<'tcx>> {
        let hint = const_eval::EvalHint::ExprTypeChecked;
        const_eval::eval_const_expr_partial(self.tcx, e, hint, None).ok().and_then(|v| {
            match v {
                // All of these contain local IDs, unsuitable for storing in MIR.
                ConstVal::Struct(_) | ConstVal::Tuple(_) |
                ConstVal::Array(..) | ConstVal::Repeat(..) |
                ConstVal::Function(_) => None,

                _ => Some(Literal::Value { value: v })
            }
        })
    }

    pub fn trait_method(&mut self,
                        trait_def_id: DefId,
                        method_name: &str,
                        self_ty: Ty<'tcx>,
                        params: Vec<Ty<'tcx>>)
                        -> (Ty<'tcx>, Literal<'tcx>) {
        let method_name = token::intern(method_name);
        let substs = Substs::new_trait(params, vec![], self_ty);
        for trait_item in self.tcx.trait_items(trait_def_id).iter() {
            match *trait_item {
                ty::ImplOrTraitItem::MethodTraitItem(ref method) => {
                    if method.name == method_name {
                        let method_ty = self.tcx.lookup_item_type(method.def_id);
                        let method_ty = method_ty.ty.subst(self.tcx, &substs);
                        return (method_ty, Literal::Item {
                            def_id: method.def_id,
                            substs: self.tcx.mk_substs(substs),
                        });
                    }
                }
                ty::ImplOrTraitItem::ConstTraitItem(..) |
                ty::ImplOrTraitItem::TypeTraitItem(..) => {}
            }
        }

        self.tcx.sess.bug(&format!("found no method `{}` in `{:?}`", method_name, trait_def_id));
    }

    pub fn num_variants(&mut self, adt_def: ty::AdtDef<'tcx>) -> usize {
        adt_def.variants.len()
    }

    pub fn all_fields(&mut self, adt_def: ty::AdtDef<'tcx>, variant_index: usize) -> Vec<Field> {
        (0..adt_def.variants[variant_index].fields.len())
            .map(Field::new)
            .collect()
    }

    pub fn needs_drop(&mut self, ty: Ty<'tcx>) -> bool {
        self.tcx.type_needs_drop_given_env(ty, &self.infcx.parameter_environment)
    }

    pub fn span_bug(&mut self, span: Span, message: &str) -> ! {
        self.tcx.sess.span_bug(span, message)
    }

    pub fn tcx(&self) -> &'a TyCtxt<'tcx> {
        self.tcx
    }
}

mod block;
mod expr;
mod pattern;
mod to_ref;
