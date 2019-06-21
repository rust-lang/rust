//! This module contains the fcuntaiontliy to convert from the wacky tcx data
//! structures into the HAIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::hair::*;
use crate::hair::util::UserAnnotatedTyHelpers;

use rustc_data_structures::indexed_vec::Idx;
use rustc::hir::def_id::DefId;
use rustc::hir::Node;
use rustc::middle::region;
use rustc::infer::InferCtxt;
use rustc::ty::subst::Subst;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{Kind, InternalSubsts};
use rustc::ty::layout::VariantIdx;
use syntax::ast;
use syntax::attr;
use syntax::symbol::{Symbol, sym};
use rustc::hir;
use crate::hair::constant::{lit_to_const, LitToConstError};

#[derive(Clone)]
pub struct Cx<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    infcx: &'a InferCtxt<'a, 'tcx>,

    pub root_lint_level: hir::HirId,
    pub param_env: ty::ParamEnv<'tcx>,

    /// Identity `InternalSubsts` for use with const-evaluation.
    pub identity_substs: &'tcx InternalSubsts<'tcx>,

    pub region_scope_tree: &'tcx region::ScopeTree,
    pub tables: &'a ty::TypeckTables<'tcx>,

    /// This is `Constness::Const` if we are compiling a `static`,
    /// `const`, or the body of a `const fn`.
    constness: hir::Constness,

    /// The `DefId` of the owner of this body.
    body_owner: DefId,

    /// What kind of body is being compiled.
    pub body_owner_kind: hir::BodyOwnerKind,

    /// Whether this constant/function needs overflow checks.
    check_overflow: bool,

    /// See field with the same name on `mir::Body`.
    control_flow_destroyed: Vec<(Span, String)>,
}

impl<'a, 'tcx> Cx<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>, src_id: hir::HirId) -> Cx<'a, 'tcx> {
        let tcx = infcx.tcx;
        let src_def_id = tcx.hir().local_def_id_from_hir_id(src_id);
        let tables = tcx.typeck_tables_of(src_def_id);
        let body_owner_kind = tcx.hir().body_owner_kind(src_id);

        let constness = match body_owner_kind {
            hir::BodyOwnerKind::Const |
            hir::BodyOwnerKind::Static(_) => hir::Constness::Const,
            hir::BodyOwnerKind::Closure |
            hir::BodyOwnerKind::Fn => hir::Constness::NotConst,
        };

        let attrs = tcx.hir().attrs(src_id);

        // Some functions always have overflow checks enabled,
        // however, they may not get codegen'd, depending on
        // the settings for the crate they are codegened in.
        let mut check_overflow = attr::contains_name(attrs, sym::rustc_inherit_overflow_checks);

        // Respect -C overflow-checks.
        check_overflow |= tcx.sess.overflow_checks();

        // Constants always need overflow checks.
        check_overflow |= constness == hir::Constness::Const;

        Cx {
            tcx,
            infcx,
            root_lint_level: src_id,
            param_env: tcx.param_env(src_def_id),
            identity_substs: InternalSubsts::identity_for_item(tcx.global_tcx(), src_def_id),
            region_scope_tree: tcx.region_scope_tree(src_def_id),
            tables,
            constness,
            body_owner: src_def_id,
            body_owner_kind,
            check_overflow,
            control_flow_destroyed: Vec::new(),
        }
    }

    pub fn control_flow_destroyed(self) -> Vec<(Span, String)> {
        self.control_flow_destroyed
    }
}

impl<'a, 'tcx> Cx<'a, 'tcx> {
    /// Normalizes `ast` into the appropriate "mirror" type.
    pub fn mirror<M: Mirror<'tcx>>(&mut self, ast: M) -> M::Output {
        ast.make_mirror(self)
    }

    pub fn usize_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.usize
    }

    pub fn usize_literal(&mut self, value: u64) -> &'tcx ty::Const<'tcx> {
        ty::Const::from_usize(self.tcx, value)
    }

    pub fn bool_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.bool
    }

    pub fn unit_ty(&mut self) -> Ty<'tcx> {
        self.tcx.mk_unit()
    }

    pub fn true_literal(&mut self) -> &'tcx ty::Const<'tcx> {
        ty::Const::from_bool(self.tcx, true)
    }

    pub fn false_literal(&mut self) -> &'tcx ty::Const<'tcx> {
        ty::Const::from_bool(self.tcx, false)
    }

    pub fn const_eval_literal(
        &mut self,
        lit: &'tcx ast::LitKind,
        ty: Ty<'tcx>,
        sp: Span,
        neg: bool,
    ) -> &'tcx ty::Const<'tcx> {
        trace!("const_eval_literal: {:#?}, {:?}, {:?}, {:?}", lit, ty, sp, neg);

        match lit_to_const(lit, self.tcx, ty, neg) {
            Ok(c) => c,
            Err(LitToConstError::UnparseableFloat) => {
                // FIXME(#31407) this is only necessary because float parsing is buggy
                self.tcx.sess.span_err(sp, "could not evaluate float literal (see issue #31407)");
                // create a dummy value and continue compiling
                Const::from_bits(self.tcx, 0, self.param_env.and(ty))
            },
            Err(LitToConstError::Reported) => {
                // create a dummy value and continue compiling
                Const::from_bits(self.tcx, 0, self.param_env.and(ty))
            }
        }
    }

    pub fn pattern_from_hir(&mut self, p: &hir::Pat) -> Pattern<'tcx> {
        let tcx = self.tcx.global_tcx();
        let p = match tcx.hir().get(p.hir_id) {
            Node::Pat(p) | Node::Binding(p) => p,
            node => bug!("pattern became {:?}", node)
        };
        Pattern::from_hir(tcx,
                          self.param_env.and(self.identity_substs),
                          self.tables(),
                          p)
    }

    pub fn trait_method(&mut self,
                        trait_def_id: DefId,
                        method_name: Symbol,
                        self_ty: Ty<'tcx>,
                        params: &[Kind<'tcx>])
                        -> (Ty<'tcx>, &'tcx ty::Const<'tcx>) {
        let substs = self.tcx.mk_substs_trait(self_ty, params);
        for item in self.tcx.associated_items(trait_def_id) {
            if item.kind == ty::AssocKind::Method && item.ident.name == method_name {
                let method_ty = self.tcx.type_of(item.def_id);
                let method_ty = method_ty.subst(self.tcx, substs);
                return (method_ty, ty::Const::zero_sized(self.tcx, method_ty));
            }
        }

        bug!("found no method `{}` in `{:?}`", method_name, trait_def_id);
    }

    pub fn all_fields(&mut self, adt_def: &ty::AdtDef, variant_index: VariantIdx) -> Vec<Field> {
        (0..adt_def.variants[variant_index].fields.len())
            .map(Field::new)
            .collect()
    }

    pub fn needs_drop(&mut self, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(self.tcx.global_tcx(), self.param_env)
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    pub fn tables(&self) -> &'a ty::TypeckTables<'tcx> {
        self.tables
    }

    pub fn check_overflow(&self) -> bool {
        self.check_overflow
    }

    pub fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.infcx.type_is_copy_modulo_regions(self.param_env, ty, span)
    }
}

impl UserAnnotatedTyHelpers<'tcx> for Cx<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx()
    }

    fn tables(&self) -> &ty::TypeckTables<'tcx> {
        self.tables()
    }
}

mod block;
mod expr;
mod to_ref;
