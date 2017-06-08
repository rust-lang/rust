// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains the code to convert from the wacky tcx data
//! structures into the hair. The `builder` is generally ignorant of
//! the tcx etc, and instead goes through the `Cx` for most of its
//! work.
//!

use hair::*;
use rustc::mir::transform::MirSource;

use rustc::middle::const_val::{ConstEvalErr, ConstVal};
use rustc_const_eval::ConstContext;
use rustc_data_structures::indexed_vec::Idx;
use rustc::hir::def_id::DefId;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::middle::region::RegionMaps;
use rustc::infer::InferCtxt;
use rustc::ty::subst::Subst;
use rustc::ty::{self, Ty, TyCtxt};
use syntax::symbol::Symbol;
use rustc::hir;
use rustc_const_math::{ConstInt, ConstUsize};
use std::rc::Rc;

#[derive(Clone)]
pub struct Cx<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub region_maps: Rc<RegionMaps>,
    pub tables: &'a ty::TypeckTables<'gcx>,

    /// This is `Constness::Const` if we are compiling a `static`,
    /// `const`, or the body of a `const fn`.
    constness: hir::Constness,

    /// What are we compiling?
    pub src: MirSource,

    /// True if this constant/function needs overflow checks.
    check_overflow: bool,
}

impl<'a, 'gcx, 'tcx> Cx<'a, 'gcx, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, src: MirSource) -> Cx<'a, 'gcx, 'tcx> {
        let constness = match src {
            MirSource::Const(_) |
            MirSource::Static(..) => hir::Constness::Const,
            MirSource::Fn(id) => {
                let fn_like = FnLikeNode::from_node(infcx.tcx.hir.get(id));
                fn_like.map_or(hir::Constness::NotConst, |f| f.constness())
            }
            MirSource::Promoted(..) => bug!(),
        };

        let tcx = infcx.tcx;
        let src_id = src.item_id();
        let src_def_id = tcx.hir.local_def_id(src_id);

        let param_env = tcx.param_env(src_def_id);
        let region_maps = tcx.region_maps(src_def_id);
        let tables = tcx.typeck_tables_of(src_def_id);

        let attrs = tcx.hir.attrs(src_id);

        // Some functions always have overflow checks enabled,
        // however, they may not get codegen'd, depending on
        // the settings for the crate they are translated in.
        let mut check_overflow = attrs.iter()
            .any(|item| item.check_name("rustc_inherit_overflow_checks"));

        // Respect -C overflow-checks.
        check_overflow |= tcx.sess.overflow_checks();

        // Constants and const fn's always need overflow checks.
        check_overflow |= constness == hir::Constness::Const;

        Cx { tcx, infcx, param_env, region_maps, tables, constness, src, check_overflow }
    }
}

impl<'a, 'gcx, 'tcx> Cx<'a, 'gcx, 'tcx> {
    /// Normalizes `ast` into the appropriate `mirror` type.
    pub fn mirror<M: Mirror<'tcx>>(&mut self, ast: M) -> M::Output {
        ast.make_mirror(self)
    }

    pub fn usize_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.usize
    }

    pub fn usize_literal(&mut self, value: u64) -> Literal<'tcx> {
        match ConstUsize::new(value, self.tcx.sess.target.uint_type) {
            Ok(val) => Literal::Value { value: ConstVal::Integral(ConstInt::Usize(val)) },
            Err(_) => bug!("usize literal out of range for target"),
        }
    }

    pub fn bool_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.bool
    }

    pub fn unit_ty(&mut self) -> Ty<'tcx> {
        self.tcx.mk_nil()
    }

    pub fn true_literal(&mut self) -> Literal<'tcx> {
        Literal::Value { value: ConstVal::Bool(true) }
    }

    pub fn false_literal(&mut self) -> Literal<'tcx> {
        Literal::Value { value: ConstVal::Bool(false) }
    }

    pub fn const_eval_literal(&mut self, e: &hir::Expr) -> Literal<'tcx> {
        let tcx = self.tcx.global_tcx();
        match ConstContext::with_tables(tcx, self.tables()).eval(e) {
            Ok(value) => Literal::Value { value: value },
            Err(s) => self.fatal_const_eval_err(&s, e.span, "expression")
        }
    }

    pub fn fatal_const_eval_err(&self,
        err: &ConstEvalErr<'tcx>,
        primary_span: Span,
        primary_kind: &str)
        -> !
    {
        err.report(self.tcx, primary_span, primary_kind);
        self.tcx.sess.abort_if_errors();
        unreachable!()
    }

    pub fn trait_method(&mut self,
                        trait_def_id: DefId,
                        method_name: &str,
                        self_ty: Ty<'tcx>,
                        params: &[Ty<'tcx>])
                        -> (Ty<'tcx>, Literal<'tcx>) {
        let method_name = Symbol::intern(method_name);
        let substs = self.tcx.mk_substs_trait(self_ty, params);
        for item in self.tcx.associated_items(trait_def_id) {
            if item.kind == ty::AssociatedKind::Method && item.name == method_name {
                let method_ty = self.tcx.type_of(item.def_id);
                let method_ty = method_ty.subst(self.tcx, substs);
                return (method_ty,
                        Literal::Value {
                            value: ConstVal::Function(item.def_id, substs),
                        });
            }
        }

        bug!("found no method `{}` in `{:?}`", method_name, trait_def_id);
    }

    pub fn num_variants(&mut self, adt_def: &ty::AdtDef) -> usize {
        adt_def.variants.len()
    }

    pub fn all_fields(&mut self, adt_def: &ty::AdtDef, variant_index: usize) -> Vec<Field> {
        (0..adt_def.variants[variant_index].fields.len())
            .map(Field::new)
            .collect()
    }

    pub fn needs_drop(&mut self, ty: Ty<'tcx>) -> bool {
        let (ty, param_env) = self.tcx.lift_to_global(&(ty, self.param_env)).unwrap_or_else(|| {
            bug!("MIR: Cx::needs_drop({:?}, {:?}) got \
                  type with inference types/regions",
                 ty, self.param_env);
        });
        ty.needs_drop(self.tcx.global_tcx(), param_env)
    }

    pub fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.tcx
    }

    pub fn tables(&self) -> &'a ty::TypeckTables<'gcx> {
        self.tables
    }

    pub fn check_overflow(&self) -> bool {
        self.check_overflow
    }
}

mod block;
mod expr;
mod to_ref;
