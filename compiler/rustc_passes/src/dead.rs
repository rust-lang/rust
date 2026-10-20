// This implements the dead-code warning pass.
// All reachable symbols are live, code called from live code is live, code with certain lint
// expectations such as `#[expect(unused)]` and `#[expect(dead_code)]` is live, and everything else
// is dead.

use std::mem;
use std::ops::ControlFlow;
use std::sync::atomic::Ordering;

use hir::def_id::{LocalDefIdMap, LocalDefIdSet};
use rustc_abi::FieldIdx;
use rustc_attr_ir::find_attr;
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{ErrorGuaranteed, MultiSpan};
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, LocalModId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, ForeignItemId, ItemId, Node, PatKind, QPath};
use rustc_lint_defs::builtin::{DEAD_CODE, DEAD_CODE_PUB_IN_BINARY};
use rustc_lint_defs::{self as lint, Lint, StableLintExpectationId};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::dead_code::{
    DeadCodeLivenessSummary, LiveSymbols, Liveness, LivenessState,
};
use rustc_middle::middle::privacy::Level;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, AssocTag, TyCtxt};
use rustc_span::{Symbol, bug, kw, span_bug};
use rustc_structures::CrateType;

use crate::diagnostics::{
    ChangeFields, DeadCodePubInBinaryNote, IgnoredDerivedImpls, MultipleDeadCodes, ParentInfo,
    UselessAssignment,
};

/// Any local definition that may call something in its body block should be explored. For example,
/// if it's a live function, then we should explore its block to check for codes that may need to
/// be marked as live.
fn should_explore(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    match tcx.def_kind(def_id) {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::Fn
        | DefKind::Const
        | DefKind::Static { .. }
        | DefKind::AssocFn
        | DefKind::AssocConst
        | DefKind::Macro(_)
        | DefKind::GlobalAsm
        | DefKind::Impl { .. }
        | DefKind::OpaqueTy
        | DefKind::AnonConst
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::Ctor(..)
        | DefKind::ForeignMod => true,

        DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::Closure
        | DefKind::SyntheticCoroutineBody
        | DefKind::TestBinderConstraints => false,
    }
}

/// Represents a def that we need to traverse in order to mark
/// all of its children as live at a specific liveness.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct WorkItem {
    id: LocalDefId,
    liveness: Liveness,
}

/// Represents an impl item whose liveness is not fully known at the moment
/// but has to be reconsidered when its dependencies change their liveness.
#[derive(Debug, Copy, Clone)]
struct UnsolvedItem {
    id: LocalDefId,
    liveness: LivenessState,
}

struct MarkSymbolVisitor<'tcx> {
    worklist: Vec<WorkItem>,
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    live_symbols: LiveSymbols,
    repr_unconditionally_treats_fields_as_live: bool,
    repr_has_repr_simd_or_scalable: bool,
    in_pat: bool,
    ignore_variant_stack: Vec<DefId>,
    // maps from ADTs to ignored derived traits (e.g. Debug and Clone)
    // and the span of their respective impl (i.e., part of the derive
    // macro)
    ignored_derived_traits: LocalDefIdMap<FxIndexSet<DefId>>,
    liveness: Liveness,
    unsolved_items: Vec<UnsolvedItem>,
}

impl<'tcx> MarkSymbolVisitor<'tcx> {
    /// Gets the type-checking results for the current body.
    /// As this will ICE if called outside bodies, only call when working with
    /// `Expr` or `Pat` nodes (they are guaranteed to be found only in bodies).
    #[track_caller]
    fn typeck_results(&self) -> &'tcx ty::TypeckResults<'tcx> {
        self.maybe_typeck_results
            .expect("`MarkSymbolVisitor::typeck_results` called outside of body")
    }

    fn push_to_worklist(&mut self, work: WorkItem) {
        let WorkItem { mut id, liveness } = work;

        // in the case of tuple struct constructors we want to check the item,
        // not the generated tuple struct constructor function
        if let DefKind::Ctor(..) = self.tcx.def_kind(id) {
            id = self.tcx.local_parent(id);
        }

        if self.live_symbols.insert(id, liveness) {
            self.worklist.push(WorkItem { id, liveness });
        }
    }

    fn check_def_id(&mut self, def_id: DefId) {
        if let Some(def_id) = def_id.as_local() {
            if should_explore(self.tcx, def_id) {
                self.push_to_worklist(WorkItem { id: def_id, liveness: self.liveness });
            }
        }
    }

    fn insert_def_id(&mut self, def_id: DefId) {
        if let Some(def_id) = def_id.as_local() {
            debug_assert!(!should_explore(self.tcx, def_id));

            self.live_symbols.insert(def_id, self.liveness);
        }
    }

    fn handle_res(&mut self, res: Res) {
        match res {
            Res::PrimTy(..) | Res::SelfCtor(..) | Res::Local(..) => {}
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), ctor_def_id) => {
                // Using a variant in patterns should not make the variant live,
                // since we can just remove the match arm that matches the pattern
                if self.in_pat {
                    return;
                }
                let variant_id = self.tcx.parent(ctor_def_id);
                let enum_id = self.tcx.parent(variant_id);
                self.check_def_id(enum_id);
                if !self.ignore_variant_stack.contains(&ctor_def_id) {
                    self.check_def_id(variant_id);
                }
            }
            Res::Def(DefKind::Variant, variant_id) => {
                // Using a variant in patterns should not make the variant live,
                // since we can just remove the match arm that matches the pattern
                if self.in_pat {
                    return;
                }
                let enum_id = self.tcx.parent(variant_id);
                self.check_def_id(enum_id);
                if !self.ignore_variant_stack.contains(&variant_id) {
                    self.check_def_id(variant_id);
                }
            }
            Res::Def(_, def_id) => self.check_def_id(def_id),
            Res::SelfTyParam { trait_: t } => self.check_def_id(t),
            Res::SelfTyAlias { alias_to: i, .. } => self.check_def_id(i),
            Res::ToolMod | Res::NonMacroAttr(..) | Res::OpenMod(..) | Res::Err => {}
        }
    }

    fn lookup_and_handle_method(&mut self, id: hir::HirId) {
        if let Some(def_id) = self.typeck_results().type_dependent_def_id(id) {
            self.check_def_id(def_id);
        } else {
            assert!(
                self.typeck_results().tainted_by_errors.is_some(),
                "no type-dependent def for method"
            );
        }
    }

    fn handle_field_access(&mut self, lhs: &hir::Expr<'_>, hir_id: hir::HirId) {
        match self.typeck_results().expr_ty_adjusted(lhs).kind() {
            ty::Adt(def, _) => {
                let index = self.typeck_results().field_index(hir_id);
                self.insert_def_id(def.non_enum_variant().fields[index].did);
            }
            ty::Tuple(..) => {}
            ty::Error(_) => {}
            kind => span_bug!(lhs.span, "named field access on non-ADT: {kind:?}"),
        }
    }

    fn handle_assign(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if self
            .typeck_results()
            .expr_adjustments(expr)
            .iter()
            .any(|adj| matches!(adj.kind, ty::adjustment::Adjust::Deref(_)))
        {
            let _ = self.visit_expr(expr);
        } else if let hir::ExprKind::Field(base, ..) = expr.kind {
            // Ignore write to field
            self.handle_assign(base);
        } else {
            let _ = self.visit_expr(expr);
        }
    }

    fn check_for_self_assign(&mut self, assign: &'tcx hir::Expr<'tcx>) {
        fn check_for_self_assign_helper<'tcx>(
            typeck_results: &'tcx ty::TypeckResults<'tcx>,
            lhs: &'tcx hir::Expr<'tcx>,
            rhs: &'tcx hir::Expr<'tcx>,
        ) -> bool {
            match (&lhs.kind, &rhs.kind) {
                (hir::ExprKind::Path(qpath_l), hir::ExprKind::Path(qpath_r)) => {
                    if let (Res::Local(id_l), Res::Local(id_r)) = (
                        typeck_results.qpath_res(qpath_l, lhs.hir_id),
                        typeck_results.qpath_res(qpath_r, rhs.hir_id),
                    ) {
                        if id_l == id_r {
                            return true;
                        }
                    }
                    return false;
                }
                (hir::ExprKind::Field(lhs_l, ident_l), hir::ExprKind::Field(lhs_r, ident_r)) => {
                    if ident_l == ident_r {
                        return check_for_self_assign_helper(typeck_results, lhs_l, lhs_r);
                    }
                    return false;
                }
                _ => {
                    return false;
                }
            }
        }

        if let hir::ExprKind::Assign(lhs, rhs, _) = assign.kind
            && check_for_self_assign_helper(self.typeck_results(), lhs, rhs)
            && !assign.span.from_expansion()
        {
            let is_field_assign = matches!(lhs.kind, hir::ExprKind::Field(..));
            self.tcx.emit_node_span_lint(
                DEAD_CODE,
                assign.hir_id,
                assign.span,
                UselessAssignment { is_field_assign, ty: self.typeck_results().expr_ty(lhs) },
            )
        }
    }

    fn handle_field_pattern_match(
        &mut self,
        lhs: &hir::Pat<'_>,
        res: Res,
        pats: &[hir::PatField<'_>],
    ) {
        let variant = match self.typeck_results().node_type(lhs.hir_id).kind() {
            ty::Adt(adt, _) => {
                // Marks the ADT live if its variant appears as the pattern,
                // considering cases when we have `let T(x) = foo()` and `fn foo<T>() -> T;`,
                // we will lose the liveness info of `T` cause we cannot mark it live when visiting `foo`.
                // Related issue: https://github.com/rust-lang/rust/issues/120770
                self.check_def_id(adt.did());
                adt.variant_of_res(res)
            }
            _ => span_bug!(lhs.span, "non-ADT in struct pattern"),
        };
        for pat in pats {
            if let PatKind::Wild = pat.pat.kind {
                continue;
            }
            let index = self.typeck_results().field_index(pat.hir_id);
            self.insert_def_id(variant.fields[index].did);
        }
    }

    fn handle_tuple_field_pattern_match(
        &mut self,
        lhs: &hir::Pat<'_>,
        res: Res,
        pats: &[hir::Pat<'_>],
        dotdot: hir::DotDotPos,
    ) {
        let variant = match self.typeck_results().node_type(lhs.hir_id).kind() {
            ty::Adt(adt, _) => {
                // Marks the ADT live if its variant appears as the pattern
                self.check_def_id(adt.did());
                adt.variant_of_res(res)
            }
            _ => {
                self.tcx.dcx().span_delayed_bug(lhs.span, "non-ADT in tuple struct pattern");
                return;
            }
        };
        let dotdot = dotdot.as_opt_usize().unwrap_or(pats.len());
        let first_n = pats.iter().enumerate().take(dotdot);
        let missing = variant.fields.len() - pats.len();
        let last_n = pats.iter().enumerate().skip(dotdot).map(|(idx, pat)| (idx + missing, pat));
        for (idx, pat) in first_n.chain(last_n) {
            if let PatKind::Wild = pat.kind {
                continue;
            }
            self.insert_def_id(variant.fields[FieldIdx::from_usize(idx)].did);
        }
    }

    fn handle_offset_of(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let indices = self
            .typeck_results()
            .offset_of_data()
            .get(expr.hir_id)
            .expect("no offset_of_data for offset_of");

        for &(current_ty, variant, field) in indices {
            match current_ty.kind() {
                ty::Adt(def, _) => {
                    let field = &def.variant(variant).fields[field];
                    self.insert_def_id(field.did);
                }
                // we don't need to mark tuple fields as live,
                // but we may need to mark subfields
                ty::Tuple(_) => {}
                _ => span_bug!(expr.span, "named field access on non-ADT"),
            }
        }
    }

    fn mark_live_symbol(
        &mut self,
        work: WorkItem,
    ) -> <MarkSymbolVisitor<'tcx> as Visitor<'tcx>>::Result {
        let WorkItem { id, liveness } = work;
        self.liveness = liveness;

        if self.tcx.is_impl_trait_in_trait(id.to_def_id()) {
            // Avoid accessing the HIR for the synthesized associated type generated for RPITITs.
            self.live_symbols.insert(id, Liveness::Live);
        } else {
            self.visit_node(self.tcx.hir_node_by_def_id(id))?;
        }
        ControlFlow::Continue(())
    }

    /// Automatically generated items marked with `rustc_trivial_field_reads`
    /// will be ignored for the purposes of dead code analysis (see PR #85200
    /// for discussion).
    fn should_ignore_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) -> bool {
        if let hir::ImplItemImplKind::Trait { .. } = impl_item.impl_kind
            && let impl_of = self.tcx.local_parent(impl_item.owner_id.def_id)
            && self.tcx.is_automatically_derived(impl_of.to_def_id())
            && let trait_ref =
                self.tcx.impl_trait_ref(impl_of).instantiate_identity().skip_norm_wip()
            && find_attr!(self.tcx, trait_ref.def_id, RustcTrivialFieldReads)
        {
            if let ty::Adt(adt_def, _) = trait_ref.self_ty().kind()
                && let Some(adt_def_id) = adt_def.did().as_local()
            {
                self.ignored_derived_traits.entry(adt_def_id).or_default().insert(trait_ref.def_id);
            }
            return true;
        }

        false
    }

    fn visit_node(
        &mut self,
        node: Node<'tcx>,
    ) -> <MarkSymbolVisitor<'tcx> as Visitor<'tcx>>::Result {
        if let Node::ImplItem(impl_item) = node
            && self.should_ignore_impl_item(impl_item)
        {
            return ControlFlow::Continue(());
        }

        let unconditionally_treated_fields_as_live =
            self.repr_unconditionally_treats_fields_as_live;
        let had_repr_simd_or_scalable = self.repr_has_repr_simd_or_scalable;
        self.repr_unconditionally_treats_fields_as_live = false;
        self.repr_has_repr_simd_or_scalable = false;
        let walk_result = match node {
            Node::Item(item) => match item.kind {
                hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) => {
                    let def = self.tcx.adt_def(item.owner_id);
                    self.repr_unconditionally_treats_fields_as_live =
                        def.repr().c() || def.repr().transparent();
                    self.repr_has_repr_simd_or_scalable =
                        def.repr().simd() || def.repr().scalable();

                    intravisit::walk_item(self, item)
                }
                hir::ItemKind::ForeignMod { .. } => ControlFlow::Continue(()),
                hir::ItemKind::Trait { items: trait_item_refs, .. } => {
                    // mark assoc ty live if the trait is live
                    for trait_item in trait_item_refs {
                        if self.tcx.def_kind(trait_item.owner_id) == DefKind::AssocTy {
                            self.check_def_id(trait_item.owner_id.to_def_id());
                        }
                    }
                    intravisit::walk_item(self, item)
                }
                _ => intravisit::walk_item(self, item),
            },
            Node::TraitItem(trait_item) => {
                // mark the trait live
                let trait_item_id = trait_item.owner_id.to_def_id();
                if let Some(trait_id) = self.tcx.trait_of_assoc(trait_item_id) {
                    self.check_def_id(trait_id);
                }
                intravisit::walk_trait_item(self, trait_item)
            }
            Node::ImplItem(impl_item) => {
                let item = self.tcx.local_parent(impl_item.owner_id.def_id);
                if let hir::ImplItemImplKind::Inherent { .. } = impl_item.impl_kind {
                    // If it's a type whose items are live, then it's live, too.
                    // This is done to handle the case where, for example, the static
                    // method of a private type is used, but the type itself is never
                    // called directly.
                    let self_ty = self.tcx.type_of(item).instantiate_identity().skip_norm_wip();
                    match *self_ty.kind() {
                        ty::Adt(def, _) => self.check_def_id(def.did()),
                        ty::Foreign(did) => self.check_def_id(did),
                        ty::Dynamic(data, ..) => {
                            if let Some(def_id) = data.principal_def_id() {
                                self.check_def_id(def_id)
                            }
                        }
                        _ => {}
                    }
                }
                intravisit::walk_impl_item(self, impl_item)
            }
            Node::ForeignItem(foreign_item) => intravisit::walk_foreign_item(self, foreign_item),
            Node::OpaqueTy(opaq) => intravisit::walk_opaque_ty(self, opaq),
            _ => ControlFlow::Continue(()),
        };
        self.repr_has_repr_simd_or_scalable = had_repr_simd_or_scalable;
        self.repr_unconditionally_treats_fields_as_live = unconditionally_treated_fields_as_live;

        walk_result
    }

    fn mark_as_used_if_union(&mut self, adt: ty::AdtDef<'tcx>, fields: &[hir::ExprField<'_>]) {
        if adt.is_union() && adt.non_enum_variant().fields.len() > 1 && adt.did().is_local() {
            for field in fields {
                let index = self.typeck_results().field_index(field.hir_id);
                self.insert_def_id(adt.non_enum_variant().fields[index].did);
            }
        }
    }

    /// Returns the maximum liveness we can currently assume for the trait impl / trait impl item
    /// `local_def_id`. It is added into the unsolved_items during `create_and_seed_worklist`. The
    /// maximum liveness is derived from a combination of the liveness of the trait itself and the
    /// implementing type.
    fn get_impl_or_impl_item_liveness(&self, local_def_id: LocalDefId) -> LivenessState {
        let (impl_block_id, trait_def_id) = match self.tcx.def_kind(local_def_id) {
            // assoc impl items of traits are live if the corresponding trait items are live
            DefKind::AssocConst | DefKind::AssocTy | DefKind::AssocFn => {
                let trait_def_id =
                    self.tcx.trait_item_of(local_def_id).and_then(|def_id| def_id.as_local());
                (self.tcx.local_parent(local_def_id), trait_def_id)
            }
            // impl items are live if the corresponding traits are live
            DefKind::Impl { of_trait: true } => {
                (local_def_id, self.tcx.impl_trait_id(local_def_id).as_local())
            }
            _ => bug!(),
        };

        let trait_liveness = match trait_def_id {
            Some(trait_def_id) => match self.live_symbols.get(trait_def_id) {
                LivenessState::Unknown => {
                    return LivenessState::Unknown;
                }
                LivenessState::Known(liveness) => liveness,
            },
            // Non-local traits are always live
            None => Liveness::Live,
        };

        let ty_liveness = if let ty::Adt(adt, _) =
            self.tcx.type_of(impl_block_id).instantiate_identity().skip_norm_wip().kind()
            && let Some(adt_def_id) = adt.did().as_local()
        {
            match self.live_symbols.get(adt_def_id) {
                LivenessState::Unknown | LivenessState::Known(Liveness::AllowExpect) => {
                    if let Some(trait_def_id) = trait_def_id
                        && has_explicit_allow_expect_dead_code(self.tcx, trait_def_id)
                    {
                        // If the trait itself is explicitly at lint level
                        // allow/expect, we check the impl block even though we do not know
                        // the liveness of the type.
                        return LivenessState::Known(Liveness::AllowExpect);
                    } else {
                        // A type being at allow/expect or unknown liveness isn't enough to check the impl,
                        // no matter the liveness of the trait.
                        return LivenessState::Unknown;
                    }
                }
                LivenessState::Known(liveness) => liveness,
            }
        } else {
            // Non ADT types are live by default
            Liveness::Live
        };

        LivenessState::Known(trait_liveness.min(ty_liveness))
    }

    fn find_now_solvable_items(&mut self) {
        let mut unsolved = std::mem::take(&mut self.unsolved_items);
        let _: Vec<_> = unsolved
            .extract_if(.., |item| match self.get_impl_or_impl_item_liveness(item.id) {
                LivenessState::Known(liveness) => {
                    let liveness = item.liveness.max(liveness);
                    self.push_to_worklist(WorkItem { id: item.id, liveness });
                    matches!(liveness, Liveness::Live)
                }
                LivenessState::Unknown { .. } => false,
            })
            .collect();
        self.unsolved_items = unsolved;
    }

    fn mark_live_symbols_and_ignored_derived_traits(&mut self) -> Result<(), ErrorGuaranteed> {
        loop {
            while let Some(item) = self.worklist.pop() {
                self.mark_live_symbol(item).continue_ok()?;
            }
            self.find_now_solvable_items();
            if self.worklist.is_empty() {
                break;
            }
        }

        Ok(())
    }
}

impl<'tcx> Visitor<'tcx> for MarkSymbolVisitor<'tcx> {
    type Result = ControlFlow<ErrorGuaranteed>;

    fn visit_nested_body(&mut self, body: hir::BodyId) -> Self::Result {
        let typeck_results = self.tcx.typeck_body(body);

        // The result shouldn't be tainted, otherwise it will cause ICE.
        if let Some(guar) = typeck_results.tainted_by_errors {
            return ControlFlow::Break(guar);
        }

        let old_maybe_typeck_results = self.maybe_typeck_results.replace(typeck_results);
        let body = self.tcx.hir_body(body);
        let result = self.visit_body(body);
        self.maybe_typeck_results = old_maybe_typeck_results;

        result
    }

    fn visit_variant_data(&mut self, def: &'tcx hir::VariantData<'tcx>) -> Self::Result {
        let tcx = self.tcx;
        let unconditionally_treat_fields_as_live = self.repr_unconditionally_treats_fields_as_live;
        let has_repr_simd_or_scalable = self.repr_has_repr_simd_or_scalable;
        let effective_visibilities = &tcx.effective_visibilities(());
        let live_fields = def.fields().iter().filter_map(|f| {
            let def_id = f.def_id;
            if unconditionally_treat_fields_as_live
                || (f.is_positional() && has_repr_simd_or_scalable)
            {
                return Some((def_id, Liveness::Live));
            }
            if !effective_visibilities.is_reachable(f.hir_id.owner.def_id) {
                return None;
            }
            if effective_visibilities.is_reachable(def_id) {
                Some((def_id, Liveness::Public))
            } else {
                None
            }
        });
        self.live_symbols.extend(live_fields);

        intravisit::walk_struct_def(self, def)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) -> Self::Result {
        match expr.kind {
            hir::ExprKind::Path(ref qpath @ QPath::TypeRelative(..)) => {
                let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                self.handle_res(res);
            }
            hir::ExprKind::MethodCall(..) => {
                self.lookup_and_handle_method(expr.hir_id);
            }
            hir::ExprKind::Field(ref lhs, ..) => {
                if self.typeck_results().opt_field_index(expr.hir_id).is_some() {
                    self.handle_field_access(lhs, expr.hir_id);
                } else {
                    self.tcx.dcx().span_delayed_bug(expr.span, "couldn't resolve index for field");
                }
            }
            hir::ExprKind::Struct(qpath, fields, _) => {
                let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                self.handle_res(res);
                if let ty::Adt(adt, _) = self.typeck_results().expr_ty(expr).kind() {
                    self.mark_as_used_if_union(*adt, fields);
                }
            }
            hir::ExprKind::Closure(cls) => {
                self.insert_def_id(cls.def_id.to_def_id());
            }
            hir::ExprKind::OffsetOf(..) => {
                self.handle_offset_of(expr);
            }
            hir::ExprKind::Assign(ref lhs, ..) => {
                self.handle_assign(lhs);
                self.check_for_self_assign(expr);
            }
            _ => (),
        }

        intravisit::walk_expr(self, expr)
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) -> Self::Result {
        // Inside the body, ignore constructions of variants
        // necessary for the pattern to match. Those construction sites
        // can't be reached unless the variant is constructed elsewhere.
        let len = self.ignore_variant_stack.len();
        self.ignore_variant_stack.extend(arm.pat.necessary_variants());
        let result = intravisit::walk_arm(self, arm);
        self.ignore_variant_stack.truncate(len);

        result
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) -> Self::Result {
        self.in_pat = true;
        match pat.kind {
            PatKind::Struct(ref path, fields, _) => {
                let res = self.typeck_results().qpath_res(path, pat.hir_id);
                self.handle_field_pattern_match(pat, res, fields);
            }
            PatKind::TupleStruct(ref qpath, fields, dotdot) => {
                let res = self.typeck_results().qpath_res(qpath, pat.hir_id);
                self.handle_tuple_field_pattern_match(pat, res, fields, dotdot);
            }
            _ => (),
        }

        let result = intravisit::walk_pat(self, pat);
        self.in_pat = false;

        result
    }

    fn visit_pat_expr(&mut self, expr: &'tcx rustc_hir::PatExpr<'tcx>) -> Self::Result {
        match &expr.kind {
            rustc_hir::PatExprKind::Path(qpath) => {
                // mark the type of variant live when meeting E::V in expr
                if let ty::Adt(adt, _) = self.typeck_results().node_type(expr.hir_id).kind() {
                    self.check_def_id(adt.did());
                }

                let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                self.handle_res(res);
            }
            _ => {}
        }
        intravisit::walk_pat_expr(self, expr)
    }

    fn visit_path(&mut self, path: &hir::Path<'tcx>, _: hir::HirId) -> Self::Result {
        self.handle_res(path.res);
        intravisit::walk_path(self, path)
    }

    fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) -> Self::Result {
        // When inline const blocks are used in pattern position, paths
        // referenced by it should be considered as used.
        let in_pat = mem::replace(&mut self.in_pat, false);

        self.live_symbols.insert(c.def_id, self.liveness);
        let result = intravisit::walk_anon_const(self, c);

        self.in_pat = in_pat;

        result
    }

    fn visit_inline_const(&mut self, c: &'tcx hir::ConstBlock) -> Self::Result {
        // When inline const blocks are used in pattern position, paths
        // referenced by it should be considered as used.
        let in_pat = mem::replace(&mut self.in_pat, false);

        self.live_symbols.insert(c.def_id, self.liveness);
        let result = intravisit::walk_inline_const(self, c);

        self.in_pat = in_pat;

        result
    }

    fn visit_trait_ref(&mut self, t: &'tcx hir::TraitRef<'tcx>) -> Self::Result {
        if let Some(trait_def_id) = t.path.res.opt_def_id()
            && let Some(segment) = t.path.segments.last()
            && let Some(args) = segment.args
        {
            for constraint in args.constraints {
                if let Some(local_def_id) = self
                    .tcx
                    .associated_items(trait_def_id)
                    .find_by_ident_and_kind(
                        self.tcx,
                        constraint.ident,
                        AssocTag::Const,
                        trait_def_id,
                    )
                    .and_then(|item| item.def_id.as_local())
                {
                    self.push_to_worklist(WorkItem { id: local_def_id, liveness: self.liveness });
                }
            }
        }

        intravisit::walk_trait_ref(self, t)
    }
}

fn has_explicit_allow_expect_dead_code(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    let hir_id = tcx.local_def_id_to_hir_id(def_id);
    let lint_level = tcx.lint_level_spec_at_node(DEAD_CODE, hir_id).level();
    matches!(lint_level, lint::Allow | lint::Expect)
}

fn get_explicit_def_id_liveness(tcx: TyCtxt<'_>, def_id: LocalDefId) -> LivenessState {
    fn has_used_like_attr(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
        tcx.def_kind(def_id).has_codegen_attrs() && {
            let cg_attrs = tcx.codegen_fn_attrs(def_id);

            // #[used], #[no_mangle], #[export_name], etc also keeps the item alive
            // forcefully, e.g., for placing it in a specific section.
            cg_attrs.contains_extern_indicator()
                || cg_attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER)
                || cg_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
        }
    }

    if has_explicit_allow_expect_dead_code(tcx, def_id) {
        LivenessState::Known(Liveness::AllowExpect)
    } else if has_used_like_attr(tcx, def_id)
        || find_attr!(tcx, def_id, Lang(..) | RustcCanonicalSymbol)
    {
        LivenessState::Known(Liveness::Live)
    } else {
        LivenessState::Unknown
    }
}

/// Examine the given definition and record it in the worklist if it should be considered live.
///
/// We want to explicitly consider as live:
/// * Item annotated with #[allow(dead_code)]
///       This is done so that if we want to suppress warnings for a
///       group of dead functions, we only have to annotate the "root".
///       For example, if both `f` and `g` are dead and `f` calls `g`,
///       then annotating `f` with `#[allow(dead_code)]` will suppress
///       warning for both `f` and `g`.
///
/// * Item annotated with #[lang=".."]
///       Lang items are always callable from elsewhere.
///
/// For trait methods and implementations of traits, we are not certain that the definitions are
/// live at this stage. We record them in `unsolved_items` for later examination.
fn maybe_record_as_seed<'tcx>(
    tcx: TyCtxt<'tcx>,
    owner_id: hir::OwnerId,
    worklist: &mut Vec<WorkItem>,
    unsolved_items: &mut Vec<UnsolvedItem>,
) {
    let explicit_liveness = get_explicit_def_id_liveness(tcx, owner_id.def_id);
    if let LivenessState::Known(liveness) = explicit_liveness {
        worklist.push(WorkItem { id: owner_id.def_id, liveness });
    }
    let mut maybe_push_unsolved = |id| {
        if matches!(
            explicit_liveness,
            LivenessState::Unknown | LivenessState::Known(Liveness::Public | Liveness::AllowExpect)
        ) {
            unsolved_items.push(UnsolvedItem { id, liveness: explicit_liveness });
        }
    };

    match tcx.def_kind(owner_id) {
        DefKind::Enum => {
            if let LivenessState::Known(liveness) = explicit_liveness {
                let adt = tcx.adt_def(owner_id);
                for variant in adt.variants().iter() {
                    worklist.push(WorkItem { id: variant.def_id.expect_local(), liveness });
                }
            }
        }
        DefKind::AssocFn | DefKind::AssocConst | DefKind::AssocTy => {
            let parent = tcx.local_parent(owner_id.def_id);
            match tcx.def_kind(parent) {
                DefKind::Impl { of_trait: false } | DefKind::Trait => {}
                DefKind::Impl { of_trait: true } => {
                    // We only care about associated items of traits,
                    // because they cannot be visited directly,
                    // so we later mark them as live if their corresponding traits
                    // or trait items and self types are both live,
                    // but inherent associated items can be visited and marked directly.
                    maybe_push_unsolved(owner_id.def_id);
                }
                _ => bug!(),
            }
        }
        DefKind::Impl { of_trait: true } if matches!(explicit_liveness, LivenessState::Unknown) => {
            maybe_push_unsolved(owner_id.def_id);
        }
        DefKind::GlobalAsm => {
            // global_asm! is always live.
            worklist.push(WorkItem { id: owner_id.def_id, liveness: Liveness::Live });
        }
        DefKind::Const => {
            if tcx.item_name(owner_id.def_id) == kw::Underscore {
                // `const _` is always live, as that syntax only exists for the side effects
                // of type checking and evaluating the constant expression, and marking them
                // as dead code would defeat that purpose.
                worklist.push(WorkItem { id: owner_id.def_id, liveness: Liveness::Live });
            }
        }
        _ => {}
    }
}

struct SeedWorklists {
    worklist: Vec<WorkItem>,
    unsolved_items: Vec<UnsolvedItem>,
}

fn create_and_seed_worklist(tcx: TyCtxt<'_>) -> SeedWorklists {
    let mut unsolved_items = Vec::new();
    let mut worklist = Vec::new();

    if let Some((def_id, _)) = tcx.entry_fn(())
        && let Some(local_def_id) = def_id.as_local()
    {
        worklist.push(WorkItem { id: local_def_id, liveness: Liveness::Live });
    }

    // Under `--test`, what `main` resolves to is the would-be entry point of a normal build,
    // so keep it live, unless a stripped user `#[rustc_main]` would have been the entry instead.
    if tcx.sess.is_test_crate()
        && !tcx.sess.removed_rustc_main_attr.load(Ordering::Relaxed)
        && let Some(main_def) = tcx.resolutions(()).main_def
        && let Some(def_id) = main_def.opt_fn_def_id()
        && let Some(local_def_id) = def_id.as_local()
    {
        worklist.push(WorkItem { id: local_def_id, liveness: Liveness::Live });
    }

    for (id, effective_vis) in tcx.effective_visibilities(()).iter() {
        if effective_vis.is_public_at_level(Level::Reachable) {
            worklist.push(WorkItem { id: *id, liveness: Liveness::Public });
        }
    }

    let crate_items = tcx.hir_crate_items(());
    for id in crate_items.owners() {
        maybe_record_as_seed(tcx, id, &mut worklist, &mut unsolved_items);
    }

    SeedWorklists { worklist, unsolved_items }
}

fn live_symbols_and_ignored_derived_traits(
    tcx: TyCtxt<'_>,
    (): (),
) -> Result<DeadCodeLivenessSummary, ErrorGuaranteed> {
    let SeedWorklists { worklist, unsolved_items } = create_and_seed_worklist(tcx);
    let mut symbol_visitor = MarkSymbolVisitor {
        live_symbols: worklist.iter().map(|item| (item.id, item.liveness)).collect(),
        worklist,
        tcx,
        maybe_typeck_results: None,
        repr_unconditionally_treats_fields_as_live: false,
        repr_has_repr_simd_or_scalable: false,
        in_pat: false,
        ignore_variant_stack: vec![],
        ignored_derived_traits: Default::default(),
        // The value doesn't matter, this will be overwritten by the first work item
        liveness: Liveness::AllowExpect,
        unsolved_items,
    };
    symbol_visitor.mark_live_symbols_and_ignored_derived_traits()?;

    Ok(DeadCodeLivenessSummary {
        live_symbols: symbol_visitor.live_symbols,
        ignored_derived_traits: symbol_visitor.ignored_derived_traits,
    })
}

struct DeadItem {
    def_id: LocalDefId,
    name: Symbol,
    level_plus: (lint::Level, Option<StableLintExpectationId>),
}

struct DeadVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    target_lint: &'static Lint,
    live_symbols: &'tcx LiveSymbols,
    ignored_derived_traits: &'tcx LocalDefIdMap<FxIndexSet<DefId>>,
    required_liveness: Liveness,
    max_fulfilling_liveness: Liveness,
}

enum ShouldWarnAboutField {
    Yes,
    No,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ReportOn {
    /// Report on something that hasn't got a proper name to refer to
    TupleField,
    /// Report on something that has got a name, which could be a field but also a method
    NamedField,
}

impl<'tcx> DeadVisitor<'tcx> {
    fn should_warn_about_field(&mut self, field: &ty::FieldDef) -> ShouldWarnAboutField {
        if self.consider_live(field.did.expect_local()) {
            return ShouldWarnAboutField::No;
        }
        let field_type = self.tcx.type_of(field.did).instantiate_identity().skip_norm_wip();
        if field_type.is_phantom_data() {
            return ShouldWarnAboutField::No;
        }
        let is_positional = field.name.as_str().starts_with(|c: char| c.is_ascii_digit());
        if is_positional
            && self
                .tcx
                .layout_of(
                    ty::TypingEnv::non_body_analysis(self.tcx, field.did)
                        .as_query_input(field_type),
                )
                .map_or(true, |layout| layout.is_zst())
        {
            return ShouldWarnAboutField::No;
        }
        ShouldWarnAboutField::Yes
    }

    fn def_lint_level_plus(
        &self,
        id: LocalDefId,
    ) -> (lint::Level, Option<StableLintExpectationId>) {
        let hir_id = self.tcx.local_def_id_to_hir_id(id);
        let level_spec = self.tcx.lint_level_spec_at_node(self.target_lint, hir_id);
        (level_spec.level(), level_spec.lint_id())
    }

    fn fulfill_dead_code_expectations(&self, def_id: LocalDefId) {
        let fulfill_if_expected = |node: DefId| {
            // Only consider local symbols where lint level carries an expectation
            // that are considered dead from the perspective of this lint, i.e. whose
            // liveness is at most `self.max_fulfilling_liveness`.
            if let Some(node) = node.as_local()
                && !self.is_unnamed_or_underscore(node)
                && self.live_symbols.at_most_as_live(node, self.max_fulfilling_liveness)
                && let (_, Some(expectation)) = self.def_lint_level_plus(node)
            {
                // Same mechanism as LintContext::fulfill_expectation.
                self.tcx.dcx().fulfill_expectation(expectation);
            }
        };

        fulfill_if_expected(def_id.to_def_id());

        match self.tcx.def_kind(def_id) {
            DefKind::Struct | DefKind::Union | DefKind::Enum => {
                let adt = self.tcx.adt_def(def_id);
                for variant in adt.variants() {
                    if variant.def_id != def_id.to_def_id() {
                        fulfill_if_expected(variant.def_id);
                    }
                    for field in &variant.fields {
                        fulfill_if_expected(field.did);
                    }
                }
            }
            DefKind::Variant => {
                let parent_enum = self.tcx.local_parent(def_id);
                let variant = self.tcx.adt_def(parent_enum).variant_with_id(def_id.to_def_id());
                //Check to see if fields carry expectation
                for field in &variant.fields {
                    fulfill_if_expected(field.did);
                }
            }
            DefKind::Trait | DefKind::Impl { .. } => {
                for &assoc_def_id in self.tcx.associated_item_def_ids(def_id) {
                    fulfill_if_expected(assoc_def_id);
                }
            }
            _ => {}
        }
    }

    fn dead_code_pub_in_binary_note(&self) -> Option<DeadCodePubInBinaryNote> {
        self.target_lint.name.eq(DEAD_CODE_PUB_IN_BINARY.name).then_some(DeadCodePubInBinaryNote)
    }

    // # Panics
    // All `dead_codes` must have the same lint level, otherwise we will intentionally ICE.
    // This is because we emit a multi-spanned lint using the lint level of the `dead_codes`'s
    // first local def id.
    // Prefer calling `Self.warn_dead_code` or `Self.warn_dead_code_grouped_by_lint_level`
    // since those methods group by lint level before calling this method.
    fn lint_at_single_level(
        &self,
        dead_codes: &[&DeadItem],
        participle: &str,
        parent_item: Option<LocalDefId>,
        report_on: ReportOn,
    ) {
        let Some(&first_item) = dead_codes.first() else { return };
        let tcx = self.tcx;

        let first_lint_level_plus = first_item.level_plus;
        assert!(dead_codes.iter().skip(1).all(|item| item.level_plus == first_lint_level_plus));

        let names: Vec<_> = dead_codes.iter().map(|item| item.name).collect();
        let spans: Vec<_> = dead_codes
            .iter()
            .map(|item| {
                let span = tcx.def_span(item.def_id);
                let ident_span = tcx.def_ident_span(item.def_id);
                // FIXME(cjgillot) this SyntaxContext manipulation does not make any sense.
                ident_span.map(|s| s.with_ctxt(span.ctxt())).unwrap_or(span)
            })
            .collect();

        let mut descr = tcx.def_descr(first_item.def_id.to_def_id());
        // `impl` blocks are "batched" and (unlike other batching) might
        // contain different kinds of associated items.
        if dead_codes.iter().any(|item| tcx.def_descr(item.def_id.to_def_id()) != descr) {
            descr = "associated item"
        }

        let num = dead_codes.len();
        let multiple = num > 6;
        let name_list = names.into();

        let parent_info = parent_item.map(|parent_item| {
            let parent_descr = tcx.def_descr(parent_item.to_def_id());
            let span = if let DefKind::Impl { .. } = tcx.def_kind(parent_item) {
                tcx.def_span(parent_item)
            } else {
                tcx.def_ident_span(parent_item).unwrap()
            };
            ParentInfo { num, descr, parent_descr, span }
        });

        let mut encl_def_id = parent_item.unwrap_or(first_item.def_id);
        // `ignored_derived_traits` is computed for the enum, not for the variants.
        if let DefKind::Variant = tcx.def_kind(encl_def_id) {
            encl_def_id = tcx.local_parent(encl_def_id);
        }

        let ignored_derived_impls =
            self.ignored_derived_traits.get(&encl_def_id).map(|ign_traits| {
                let trait_list = ign_traits
                    .iter()
                    .map(|trait_id| self.tcx.item_name(*trait_id))
                    .collect::<Vec<_>>();
                let trait_list_len = trait_list.len();
                IgnoredDerivedImpls {
                    name: self.tcx.item_name(encl_def_id.to_def_id()),
                    trait_list: trait_list.into(),
                    trait_list_len,
                }
            });

        let diag = match report_on {
            ReportOn::TupleField => {
                let tuple_fields = if let Some(parent_id) = parent_item
                    && let node = tcx.hir_node_by_def_id(parent_id)
                    && let hir::Node::Item(hir::Item {
                        kind: hir::ItemKind::Struct(_, _, hir::VariantData::Tuple(fields, _, _)),
                        ..
                    }) = node
                {
                    *fields
                } else {
                    &[]
                };

                let trailing_tuple_fields = if tuple_fields.len() >= dead_codes.len() {
                    LocalDefIdSet::from_iter(
                        tuple_fields
                            .iter()
                            .skip(tuple_fields.len() - dead_codes.len())
                            .map(|f| f.def_id),
                    )
                } else {
                    LocalDefIdSet::default()
                };

                let fields_suggestion =
                    // Suggest removal if all tuple fields are at the end.
                    // Otherwise suggest removal or changing to unit type
                    if dead_codes.iter().all(|dc| trailing_tuple_fields.contains(&dc.def_id)) {
                        ChangeFields::Remove { num }
                    } else {
                        ChangeFields::ChangeToUnitTypeOrRemove { num, spans: spans.clone() }
                    };

                MultipleDeadCodes::UnusedTupleStructFields {
                    multiple,
                    num,
                    descr,
                    participle,
                    name_list,
                    dead_code_pub_in_binary_note: self.dead_code_pub_in_binary_note(),
                    change_fields_suggestion: fields_suggestion,
                    parent_info,
                    ignored_derived_impls,
                }
            }
            ReportOn::NamedField => {
                let enum_variants_with_same_name = dead_codes
                    .iter()
                    .filter_map(|dead_item| {
                        if let DefKind::AssocFn | DefKind::AssocConst =
                            tcx.def_kind(dead_item.def_id)
                            && let impl_did = tcx.local_parent(dead_item.def_id)
                            && let DefKind::Impl { of_trait: false } = tcx.def_kind(impl_did)
                            && let ty::Adt(maybe_enum, _) =
                                tcx.type_of(impl_did).instantiate_identity().skip_norm_wip().kind()
                            && maybe_enum.is_enum()
                            && let Some(variant) =
                                maybe_enum.variants().iter().find(|i| i.name == dead_item.name)
                        {
                            Some(crate::diagnostics::EnumVariantSameName {
                                dead_descr: tcx.def_descr(dead_item.def_id.to_def_id()),
                                dead_name: dead_item.name,
                                variant_span: tcx.def_span(variant.def_id),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                MultipleDeadCodes::DeadCodes {
                    multiple,
                    num,
                    descr,
                    participle,
                    name_list,
                    dead_code_pub_in_binary_note: self.dead_code_pub_in_binary_note(),
                    parent_info,
                    ignored_derived_impls,
                    enum_variants_with_same_name,
                }
            }
        };

        let hir_id = tcx.local_def_id_to_hir_id(first_item.def_id);
        self.tcx.emit_node_span_lint(self.target_lint, hir_id, MultiSpan::from_spans(spans), diag);
    }

    fn warn_multiple(
        &self,
        def_id: LocalDefId,
        participle: &str,
        dead_codes: Vec<DeadItem>,
        report_on: ReportOn,
    ) {
        let mut dead_codes = dead_codes
            .iter()
            .filter(|v| !v.name.as_str().starts_with('_'))
            .collect::<Vec<&DeadItem>>();
        if dead_codes.is_empty() {
            return;
        }
        // FIXME: `dead_codes` should probably be morally equivalent to
        // `IndexMap<(Level, StableLintExpectationId), (DefId, Symbol)>`
        dead_codes.sort_by_key(|v| v.level_plus.0);
        for group in dead_codes.chunk_by(|a, b| a.level_plus == b.level_plus) {
            self.lint_at_single_level(&group, participle, Some(def_id), report_on);
        }
    }

    fn warn_dead_code(&mut self, id: LocalDefId, participle: &str) {
        let item = DeadItem {
            def_id: id,
            name: self.tcx.item_name(id.to_def_id()),
            level_plus: self.def_lint_level_plus(id),
        };
        self.lint_at_single_level(&[&item], participle, None, ReportOn::NamedField);
    }

    fn check_definition(&mut self, def_id: LocalDefId) {
        if self.is_unnamed_or_underscore(def_id) || self.consider_live(def_id) {
            return;
        }
        match self.tcx.def_kind(def_id) {
            DefKind::AssocConst
            | DefKind::AssocTy
            | DefKind::AssocFn
            | DefKind::Fn
            | DefKind::Static { .. }
            | DefKind::Const
            | DefKind::TyAlias
            | DefKind::Enum
            | DefKind::Union
            | DefKind::ForeignTy
            | DefKind::Trait => self.warn_dead_code(def_id, "used"),
            DefKind::Struct => self.warn_dead_code(def_id, "constructed"),
            DefKind::Variant | DefKind::Field => bug!("should be handled specially"),
            _ => {}
        }
    }

    fn is_unnamed_or_underscore(&self, def_id: LocalDefId) -> bool {
        // if we cannot get a name for the item, then we just assume that it is
        // live. I mean, we can't really emit a lint.
        let Some(name) = self.tcx.opt_item_name(def_id.to_def_id()) else {
            return true;
        };

        name.as_str().starts_with('_')
    }

    fn consider_live(&self, did: LocalDefId) -> bool {
        self.live_symbols.at_least_as_live(did, self.required_liveness)
    }
}

fn check_mod_deathness(tcx: TyCtxt<'_>, module: LocalModId) {
    let Ok(DeadCodeLivenessSummary { live_symbols, ignored_derived_traits }) =
        tcx.live_symbols_and_ignored_derived_traits(()).as_ref()
    else {
        return;
    };

    let module_items = tcx.hir_module_items(module);

    if tcx.crate_types().contains(&CrateType::Executable) {
        let is_unused_pub = |def_id: LocalDefId| {
            tcx.effective_visibilities(()).is_public_at_level(def_id, Level::Reachable)
                && !live_symbols.is_live(def_id)
        };

        lint_dead_codes(
            tcx,
            DEAD_CODE_PUB_IN_BINARY,
            module,
            &live_symbols,
            &ignored_derived_traits,
            module_items.free_items().filter(|free_item| is_unused_pub(free_item.owner_id.def_id)),
            module_items
                .foreign_items()
                .filter(|foreign_item| is_unused_pub(foreign_item.owner_id.def_id)),
            // For this lint, only properly `Live` items are exempt
            Liveness::Live,
            // And `#[expect(dead_code_pub_in_binary)]` requires items to be at most `Public`
            Liveness::Public,
        );
    }

    lint_dead_codes(
        tcx,
        DEAD_CODE,
        module,
        &live_symbols,
        &ignored_derived_traits,
        module_items.free_items(),
        module_items.foreign_items(),
        // For this lint, all items with at least `AllowExpect` are exempt
        Liveness::AllowExpect,
        // And `#[expect(dead_code)]` requires items to be at most `AllowExpect`
        Liveness::AllowExpect,
    );
}

fn lint_dead_codes<'tcx>(
    tcx: TyCtxt<'tcx>,
    target_lint: &'static Lint,
    module: LocalModId,
    live_symbols: &'tcx LiveSymbols,
    ignored_derived_traits: &'tcx LocalDefIdMap<FxIndexSet<DefId>>,
    free_items: impl Iterator<Item = ItemId>,
    foreign_items: impl Iterator<Item = ForeignItemId>,
    required_liveness: Liveness,
    max_fulfilling_liveness: Liveness,
) {
    let mut visitor = DeadVisitor {
        tcx,
        target_lint,
        live_symbols,
        ignored_derived_traits,
        required_liveness,
        max_fulfilling_liveness,
    };
    for item in free_items {
        let def_kind = tcx.def_kind(item.owner_id);
        // use stmts are handled by `unused_imports`, so skip them
        if def_kind == DefKind::Use {
            continue;
        }
        visitor.fulfill_dead_code_expectations(item.owner_id.def_id);

        let mut dead_codes = Vec::new();
        // Only diagnose unused assoc items in inherent impl and used trait.
        // For unused assoc items in impls of trait,
        // we have diagnosed them in the trait if they are unused,
        // for unused assoc items in unused trait,
        // we have diagnosed the unused trait.
        if def_kind == (DefKind::Impl { of_trait: false })
            || (def_kind == DefKind::Trait && visitor.consider_live(item.owner_id.def_id))
        {
            for &def_id in tcx.associated_item_def_ids(item.owner_id.def_id) {
                if let Some(local_def_id) = def_id.as_local()
                    && !visitor.is_unnamed_or_underscore(local_def_id)
                    && !visitor.consider_live(local_def_id)
                {
                    let name = tcx.item_name(def_id);
                    let level_plus = visitor.def_lint_level_plus(local_def_id);
                    dead_codes.push(DeadItem { def_id: local_def_id, name, level_plus });
                }
            }
        }
        if !dead_codes.is_empty() {
            visitor.warn_multiple(item.owner_id.def_id, "used", dead_codes, ReportOn::NamedField);
        }

        if !visitor.consider_live(item.owner_id.def_id) {
            let parent = tcx.local_parent(item.owner_id.def_id);
            if parent != module.to_local_def_id() && !visitor.consider_live(parent) {
                continue;
            }
            visitor.check_definition(item.owner_id.def_id);
            continue;
        }

        if let DefKind::Struct | DefKind::Union | DefKind::Enum = def_kind {
            let adt = tcx.adt_def(item.owner_id);
            let mut dead_variants = Vec::new();

            for variant in adt.variants() {
                let def_id = variant.def_id.expect_local();
                if !visitor.consider_live(def_id) {
                    // Record to group diagnostics.
                    let level_plus = visitor.def_lint_level_plus(def_id);
                    dead_variants.push(DeadItem { def_id, name: variant.name, level_plus });
                    continue;
                }

                let is_positional = variant.fields.raw.first().is_some_and(|field| {
                    field.name.as_str().starts_with(|c: char| c.is_ascii_digit())
                });
                let report_on =
                    if is_positional { ReportOn::TupleField } else { ReportOn::NamedField };
                let dead_fields = variant
                    .fields
                    .iter()
                    .filter_map(|field| {
                        let def_id = field.did.expect_local();
                        if let ShouldWarnAboutField::Yes = visitor.should_warn_about_field(field) {
                            let level_plus = visitor.def_lint_level_plus(def_id);
                            Some(DeadItem { def_id, name: field.name, level_plus })
                        } else {
                            None
                        }
                    })
                    .collect();
                visitor.warn_multiple(def_id, "read", dead_fields, report_on);
            }

            visitor.warn_multiple(
                item.owner_id.def_id,
                "constructed",
                dead_variants,
                ReportOn::NamedField,
            );
        }
    }

    for foreign_item in foreign_items {
        visitor.fulfill_dead_code_expectations(foreign_item.owner_id.def_id);
        visitor.check_definition(foreign_item.owner_id.def_id);
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers =
        Providers { live_symbols_and_ignored_derived_traits, check_mod_deathness, ..*providers };
}
