//! This module contains the fcuntaiontliy to convert from the wacky tcx data
//! structures into the HAIR. The `builder` is generally ignorant of the tcx,
//! etc., and instead goes through the `Cx` for most of its work.

use crate::hair::util::UserAnnotatedTyHelpers;
use crate::hair::*;

use rustc::middle::region;
use rustc::mir::interpret::{LitToConstError, LitToConstInput};
use rustc::ty::layout::VariantIdx;
use rustc::ty::subst::Subst;
use rustc::ty::subst::{GenericArg, InternalSubsts};
use rustc::ty::{self, Ty, TyCtxt};
use rustc_ast::ast;
use rustc_ast::attr;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::Node;
use rustc_index::vec::Idx;
use rustc_infer::infer::InferCtxt;
use rustc_span::symbol::{sym, Symbol};
use rustc_trait_selection::infer::InferCtxtExt;

#[derive(Clone)]
crate struct Cx<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    infcx: &'a InferCtxt<'a, 'tcx>,

    crate root_lint_level: hir::HirId,
    crate param_env: ty::ParamEnv<'tcx>,

    /// Identity `InternalSubsts` for use with const-evaluation.
    crate identity_substs: &'tcx InternalSubsts<'tcx>,

    crate region_scope_tree: &'tcx region::ScopeTree,
    crate tables: &'a ty::TypeckTables<'tcx>,

    /// This is `Constness::Const` if we are compiling a `static`,
    /// `const`, or the body of a `const fn`.
    constness: hir::Constness,

    /// The `DefId` of the owner of this body.
    body_owner: DefId,

    /// What kind of body is being compiled.
    crate body_owner_kind: hir::BodyOwnerKind,

    /// Whether this constant/function needs overflow checks.
    check_overflow: bool,

    /// See field with the same name on `mir::Body`.
    control_flow_destroyed: Vec<(Span, String)>,
}

impl<'a, 'tcx> Cx<'a, 'tcx> {
    crate fn new(infcx: &'a InferCtxt<'a, 'tcx>, src_id: hir::HirId) -> Cx<'a, 'tcx> {
        let tcx = infcx.tcx;
        let src_def_id = tcx.hir().local_def_id(src_id);
        let tables = tcx.typeck_tables_of(src_def_id);
        let body_owner_kind = tcx.hir().body_owner_kind(src_id);

        let constness = match body_owner_kind {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => hir::Constness::Const,
            hir::BodyOwnerKind::Closure | hir::BodyOwnerKind::Fn => hir::Constness::NotConst,
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
            identity_substs: InternalSubsts::identity_for_item(tcx, src_def_id),
            region_scope_tree: tcx.region_scope_tree(src_def_id),
            tables,
            constness,
            body_owner: src_def_id,
            body_owner_kind,
            check_overflow,
            control_flow_destroyed: Vec::new(),
        }
    }

    crate fn control_flow_destroyed(self) -> Vec<(Span, String)> {
        self.control_flow_destroyed
    }
}

impl<'a, 'tcx> Cx<'a, 'tcx> {
    /// Normalizes `ast` into the appropriate "mirror" type.
    crate fn mirror<M: Mirror<'tcx>>(&mut self, ast: M) -> M::Output {
        ast.make_mirror(self)
    }

    crate fn usize_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.usize
    }

    crate fn usize_literal(&mut self, value: u64) -> &'tcx ty::Const<'tcx> {
        ty::Const::from_usize(self.tcx, value)
    }

    crate fn bool_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.bool
    }

    crate fn unit_ty(&mut self) -> Ty<'tcx> {
        self.tcx.mk_unit()
    }

    crate fn true_literal(&mut self) -> &'tcx ty::Const<'tcx> {
        ty::Const::from_bool(self.tcx, true)
    }

    crate fn false_literal(&mut self) -> &'tcx ty::Const<'tcx> {
        ty::Const::from_bool(self.tcx, false)
    }

    crate fn const_eval_literal(
        &mut self,
        lit: &'tcx ast::LitKind,
        ty: Ty<'tcx>,
        sp: Span,
        neg: bool,
    ) -> &'tcx ty::Const<'tcx> {
        trace!("const_eval_literal: {:#?}, {:?}, {:?}, {:?}", lit, ty, sp, neg);

        match self.tcx.at(sp).lit_to_const(LitToConstInput { lit, ty, neg }) {
            Ok(c) => c,
            Err(LitToConstError::UnparseableFloat) => {
                // FIXME(#31407) this is only necessary because float parsing is buggy
                self.tcx.sess.span_err(sp, "could not evaluate float literal (see issue #31407)");
                // create a dummy value and continue compiling
                Const::from_bits(self.tcx, 0, self.param_env.and(ty))
            }
            Err(LitToConstError::Reported) => {
                // create a dummy value and continue compiling
                Const::from_bits(self.tcx, 0, self.param_env.and(ty))
            }
            Err(LitToConstError::TypeError) => bug!("const_eval_literal: had type error"),
        }
    }

    crate fn pattern_from_hir(&mut self, p: &hir::Pat<'_>) -> Pat<'tcx> {
        let p = match self.tcx.hir().get(p.hir_id) {
            Node::Pat(p) | Node::Binding(p) => p,
            node => bug!("pattern became {:?}", node),
        };
        Pat::from_hir(self.tcx, self.param_env, self.tables(), p)
    }

    crate fn trait_method(
        &mut self,
        trait_def_id: DefId,
        method_name: Symbol,
        self_ty: Ty<'tcx>,
        params: &[GenericArg<'tcx>],
    ) -> &'tcx ty::Const<'tcx> {
        let substs = self.tcx.mk_substs_trait(self_ty, params);

        // The unhygienic comparison here is acceptable because this is only
        // used on known traits.
        let item = self
            .tcx
            .associated_items(trait_def_id)
            .filter_by_name_unhygienic(method_name)
            .find(|item| item.kind == ty::AssocKind::Method)
            .expect("trait method not found");

        let method_ty = self.tcx.type_of(item.def_id);
        let method_ty = method_ty.subst(self.tcx, substs);
        ty::Const::zero_sized(self.tcx, method_ty)
    }

    crate fn all_fields(&mut self, adt_def: &ty::AdtDef, variant_index: VariantIdx) -> Vec<Field> {
        (0..adt_def.variants[variant_index].fields.len()).map(Field::new).collect()
    }

    crate fn needs_drop(&mut self, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(self.tcx, self.param_env)
    }

    crate fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    crate fn tables(&self) -> &'a ty::TypeckTables<'tcx> {
        self.tables
    }

    crate fn check_overflow(&self) -> bool {
        self.check_overflow
    }

    crate fn type_is_copy_modulo_regions(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.infcx.type_is_copy_modulo_regions(self.param_env, ty, span)
    }
}

impl<'tcx> UserAnnotatedTyHelpers<'tcx> for Cx<'_, 'tcx> {
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
