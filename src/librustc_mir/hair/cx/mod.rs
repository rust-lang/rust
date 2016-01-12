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
use rustc::middle::infer::InferCtxt;
use rustc::middle::ty::{self, Ty};
use syntax::codemap::Span;
use syntax::parse::token;
use rustc_front::hir;

#[derive(Copy, Clone)]
pub struct Cx<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
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

    pub fn usize_literal(&mut self, value: usize) -> Literal<'tcx> {
        Literal::Value { value: ConstVal::Uint(value as u64) }
    }

    pub fn bool_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.bool
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
        const_eval::eval_const_expr_partial(self.tcx, e, hint, None)
            .ok()
            .map(|v| Literal::Value { value: v })
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

    pub fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.tcx
    }
}

mod block;
mod expr;
mod pattern;
mod to_ref;
