// This implements the dead-code warning pass.
// All reachable symbols are live, code called from live code is live, code with certain lint
// expectations such as `#[expect(unused)]` and `#[expect(dead_code)]` is live, and everything else
// is dead.

use std::mem;

use hir::ItemKind;
use hir::def_id::{LocalDefIdMap, LocalDefIdSet};
use rustc_abi::FieldIdx;
use rustc_data_structures::unord::UnordSet;
use rustc_errors::MultiSpan;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId, LocalModDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, Node, PatKind, TyKind};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::privacy::Level;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::lint::builtin::DEAD_CODE;
use rustc_session::lint::{self, LintExpectationId};
use rustc_span::{Symbol, sym};

use crate::errors::{
    ChangeFields, IgnoredDerivedImpls, MultipleDeadCodes, ParentInfo, UselessAssignment,
};

// Any local node that may call something in its body block should be
// explored. For example, if it's a live Node::Item that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    matches!(
        tcx.hir_node_by_def_id(def_id),
        Node::Item(..)
            | Node::ImplItem(..)
            | Node::ForeignItem(..)
            | Node::TraitItem(..)
            | Node::Variant(..)
            | Node::AnonConst(..)
            | Node::OpaqueTy(..)
    )
}

fn ty_ref_to_pub_struct(tcx: TyCtxt<'_>, ty: &hir::Ty<'_>) -> bool {
    if let TyKind::Path(hir::QPath::Resolved(_, path)) = ty.kind
        && let Res::Def(def_kind, def_id) = path.res
        && def_id.is_local()
        && matches!(def_kind, DefKind::Struct | DefKind::Enum | DefKind::Union)
    {
        tcx.visibility(def_id).is_public()
    } else {
        true
    }
}

/// Determine if a work from the worklist is coming from a `#[allow]`
/// or a `#[expect]` of `dead_code`
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum ComesFromAllowExpect {
    Yes,
    No,
}

struct MarkSymbolVisitor<'tcx> {
    worklist: Vec<(LocalDefId, ComesFromAllowExpect)>,
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    live_symbols: LocalDefIdSet,
    repr_unconditionally_treats_fields_as_live: bool,
    repr_has_repr_simd: bool,
    in_pat: bool,
    ignore_variant_stack: Vec<DefId>,
    // maps from tuple struct constructors to tuple struct items
    struct_constructors: LocalDefIdMap<LocalDefId>,
    // maps from ADTs to ignored derived traits (e.g. Debug and Clone)
    // and the span of their respective impl (i.e., part of the derive
    // macro)
    ignored_derived_traits: LocalDefIdMap<Vec<(DefId, DefId)>>,
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

    fn check_def_id(&mut self, def_id: DefId) {
        if let Some(def_id) = def_id.as_local() {
            if should_explore(self.tcx, def_id) || self.struct_constructors.contains_key(&def_id) {
                self.worklist.push((def_id, ComesFromAllowExpect::No));
            }
            self.live_symbols.insert(def_id);
        }
    }

    fn insert_def_id(&mut self, def_id: DefId) {
        if let Some(def_id) = def_id.as_local() {
            debug_assert!(!should_explore(self.tcx, def_id));
            self.live_symbols.insert(def_id);
        }
    }

    fn handle_res(&mut self, res: Res) {
        match res {
            Res::Def(DefKind::Const | DefKind::AssocConst | DefKind::TyAlias, def_id) => {
                self.check_def_id(def_id);
            }
            _ if self.in_pat => {}
            Res::PrimTy(..) | Res::SelfCtor(..) | Res::Local(..) => {}
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), ctor_def_id) => {
                let variant_id = self.tcx.parent(ctor_def_id);
                let enum_id = self.tcx.parent(variant_id);
                self.check_def_id(enum_id);
                if !self.ignore_variant_stack.contains(&ctor_def_id) {
                    self.check_def_id(variant_id);
                }
            }
            Res::Def(DefKind::Variant, variant_id) => {
                let enum_id = self.tcx.parent(variant_id);
                self.check_def_id(enum_id);
                if !self.ignore_variant_stack.contains(&variant_id) {
                    self.check_def_id(variant_id);
                }
            }
            Res::Def(_, def_id) => self.check_def_id(def_id),
            Res::SelfTyParam { trait_: t } => self.check_def_id(t),
            Res::SelfTyAlias { alias_to: i, .. } => self.check_def_id(i),
            Res::ToolMod | Res::NonMacroAttr(..) | Res::Err => {}
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

    #[allow(dead_code)] // FIXME(81658): should be used + lint reinstated after #83171 relands.
    fn handle_assign(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if self
            .typeck_results()
            .expr_adjustments(expr)
            .iter()
            .any(|adj| matches!(adj.kind, ty::adjustment::Adjust::Deref(_)))
        {
            self.visit_expr(expr);
        } else if let hir::ExprKind::Field(base, ..) = expr.kind {
            // Ignore write to field
            self.handle_assign(base);
        } else {
            self.visit_expr(expr);
        }
    }

    #[allow(dead_code)] // FIXME(81658): should be used + lint reinstated after #83171 relands.
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
                lint::builtin::DEAD_CODE,
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
            ty::Adt(adt, _) => adt.variant_of_res(res),
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
            ty::Adt(adt, _) => adt.variant_of_res(res),
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
        let data = self.typeck_results().offset_of_data();
        let &(container, ref indices) =
            data.get(expr.hir_id).expect("no offset_of_data for offset_of");

        let body_did = self.typeck_results().hir_owner.to_def_id();
        let typing_env = ty::TypingEnv::non_body_analysis(self.tcx, body_did);

        let mut current_ty = container;

        for &(variant, field) in indices {
            match current_ty.kind() {
                ty::Adt(def, args) => {
                    let field = &def.variant(variant).fields[field];

                    self.insert_def_id(field.did);
                    let field_ty = field.ty(self.tcx, args);

                    current_ty = self.tcx.normalize_erasing_regions(typing_env, field_ty);
                }
                // we don't need to mark tuple fields as live,
                // but we may need to mark subfields
                ty::Tuple(tys) => {
                    current_ty =
                        self.tcx.normalize_erasing_regions(typing_env, tys[field.as_usize()]);
                }
                _ => span_bug!(expr.span, "named field access on non-ADT"),
            }
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = UnordSet::default();
        while let Some(work) = self.worklist.pop() {
            if !scanned.insert(work) {
                continue;
            }

            let (id, comes_from_allow_expect) = work;

            // Avoid accessing the HIR for the synthesized associated type generated for RPITITs.
            if self.tcx.is_impl_trait_in_trait(id.to_def_id()) {
                self.live_symbols.insert(id);
                continue;
            }

            // in the case of tuple struct constructors we want to check the item, not the generated
            // tuple struct constructor function
            let id = self.struct_constructors.get(&id).copied().unwrap_or(id);

            // When using `#[allow]` or `#[expect]` of `dead_code`, we do a QOL improvement
            // by declaring fn calls, statics, ... within said items as live, as well as
            // the item itself, although technically this is not the case.
            //
            // This means that the lint for said items will never be fired.
            //
            // This doesn't make any difference for the item declared with `#[allow]`, as
            // the lint firing will be a nop, as it will be silenced by the `#[allow]` of
            // the item.
            //
            // However, for `#[expect]`, the presence or absence of the lint is relevant,
            // so we don't add it to the list of live symbols when it comes from a
            // `#[expect]`. This means that we will correctly report an item as live or not
            // for the `#[expect]` case.
            //
            // Note that an item can and will be duplicated on the worklist with different
            // `ComesFromAllowExpect`, particularly if it was added from the
            // `effective_visibilities` query or from the `#[allow]`/`#[expect]` checks,
            // this "duplication" is essential as otherwise a function with `#[expect]`
            // called from a `pub fn` may be falsely reported as not live, falsely
            // triggering the `unfulfilled_lint_expectations` lint.
            if comes_from_allow_expect != ComesFromAllowExpect::Yes {
                self.live_symbols.insert(id);
            }
            self.visit_node(self.tcx.hir_node_by_def_id(id));
        }
    }

    /// Automatically generated items marked with `rustc_trivial_field_reads`
    /// will be ignored for the purposes of dead code analysis (see PR #85200
    /// for discussion).
    fn should_ignore_item(&mut self, def_id: DefId) -> bool {
        if let Some(impl_of) = self.tcx.impl_of_method(def_id) {
            if !self.tcx.is_automatically_derived(impl_of) {
                return false;
            }

            // don't ignore impls for Enums and pub Structs whose methods don't have self receiver,
            // cause external crate may call such methods to construct values of these types
            if let Some(local_impl_of) = impl_of.as_local()
                && let Some(local_def_id) = def_id.as_local()
                && let Some(fn_sig) =
                    self.tcx.hir_fn_sig_by_hir_id(self.tcx.local_def_id_to_hir_id(local_def_id))
                && matches!(fn_sig.decl.implicit_self, hir::ImplicitSelfKind::None)
                && let TyKind::Path(hir::QPath::Resolved(_, path)) =
                    self.tcx.hir_expect_item(local_impl_of).expect_impl().self_ty.kind
                && let Res::Def(def_kind, did) = path.res
            {
                match def_kind {
                    // for example, #[derive(Default)] pub struct T(i32);
                    // external crate can call T::default() to construct T,
                    // so that don't ignore impl Default for pub Enum and Structs
                    DefKind::Struct | DefKind::Union if self.tcx.visibility(did).is_public() => {
                        return false;
                    }
                    // don't ignore impl Default for Enums,
                    // cause we don't know which variant is constructed
                    DefKind::Enum => return false,
                    _ => (),
                };
            }

            if let Some(trait_of) = self.tcx.trait_id_of_impl(impl_of)
                && self.tcx.has_attr(trait_of, sym::rustc_trivial_field_reads)
            {
                let trait_ref = self.tcx.impl_trait_ref(impl_of).unwrap().instantiate_identity();
                if let ty::Adt(adt_def, _) = trait_ref.self_ty().kind()
                    && let Some(adt_def_id) = adt_def.did().as_local()
                {
                    self.ignored_derived_traits
                        .entry(adt_def_id)
                        .or_default()
                        .push((trait_of, impl_of));
                }
                return true;
            }
        }

        false
    }

    fn visit_node(&mut self, node: Node<'tcx>) {
        if let Node::ImplItem(hir::ImplItem { owner_id, .. }) = node
            && self.should_ignore_item(owner_id.to_def_id())
        {
            return;
        }

        let unconditionally_treated_fields_as_live =
            self.repr_unconditionally_treats_fields_as_live;
        let had_repr_simd = self.repr_has_repr_simd;
        self.repr_unconditionally_treats_fields_as_live = false;
        self.repr_has_repr_simd = false;
        match node {
            Node::Item(item) => match item.kind {
                hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) => {
                    let def = self.tcx.adt_def(item.owner_id);
                    self.repr_unconditionally_treats_fields_as_live =
                        def.repr().c() || def.repr().transparent();
                    self.repr_has_repr_simd = def.repr().simd();

                    intravisit::walk_item(self, item)
                }
                hir::ItemKind::ForeignMod { .. } => {}
                hir::ItemKind::Trait(..) => {
                    for &impl_def_id in self.tcx.local_trait_impls(item.owner_id.def_id) {
                        if let ItemKind::Impl(impl_ref) = self.tcx.hir_expect_item(impl_def_id).kind
                        {
                            // skip items
                            // mark dependent traits live
                            intravisit::walk_generics(self, impl_ref.generics);
                            // mark dependent parameters live
                            intravisit::walk_path(self, impl_ref.of_trait.unwrap().path);
                        }
                    }

                    intravisit::walk_item(self, item)
                }
                _ => intravisit::walk_item(self, item),
            },
            Node::TraitItem(trait_item) => {
                // mark corresponding ImplTerm live
                let trait_item_id = trait_item.owner_id.to_def_id();
                if let Some(trait_id) = self.tcx.trait_of_item(trait_item_id) {
                    // mark the trait live
                    self.check_def_id(trait_id);

                    for impl_id in self.tcx.all_impls(trait_id) {
                        if let Some(local_impl_id) = impl_id.as_local()
                            && let ItemKind::Impl(impl_ref) =
                                self.tcx.hir_expect_item(local_impl_id).kind
                        {
                            if !matches!(trait_item.kind, hir::TraitItemKind::Type(..))
                                && !ty_ref_to_pub_struct(self.tcx, impl_ref.self_ty)
                            {
                                // skip methods of private ty,
                                // they would be solved in `solve_rest_impl_items`
                                continue;
                            }

                            // mark self_ty live
                            intravisit::walk_unambig_ty(self, impl_ref.self_ty);
                            if let Some(&impl_item_id) =
                                self.tcx.impl_item_implementor_ids(impl_id).get(&trait_item_id)
                            {
                                self.check_def_id(impl_item_id);
                            }
                        }
                    }
                }
                intravisit::walk_trait_item(self, trait_item);
            }
            Node::ImplItem(impl_item) => {
                let item = self.tcx.local_parent(impl_item.owner_id.def_id);
                if self.tcx.impl_trait_ref(item).is_none() {
                    //// If it's a type whose items are live, then it's live, too.
                    //// This is done to handle the case where, for example, the static
                    //// method of a private type is used, but the type itself is never
                    //// called directly.
                    let self_ty = self.tcx.type_of(item).instantiate_identity();
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
                intravisit::walk_impl_item(self, impl_item);
            }
            Node::ForeignItem(foreign_item) => {
                intravisit::walk_foreign_item(self, foreign_item);
            }
            Node::OpaqueTy(opaq) => intravisit::walk_opaque_ty(self, opaq),
            _ => {}
        }
        self.repr_has_repr_simd = had_repr_simd;
        self.repr_unconditionally_treats_fields_as_live = unconditionally_treated_fields_as_live;
    }

    fn mark_as_used_if_union(&mut self, adt: ty::AdtDef<'tcx>, fields: &[hir::ExprField<'_>]) {
        if adt.is_union() && adt.non_enum_variant().fields.len() > 1 && adt.did().is_local() {
            for field in fields {
                let index = self.typeck_results().field_index(field.hir_id);
                self.insert_def_id(adt.non_enum_variant().fields[index].did);
            }
        }
    }

    fn solve_rest_impl_items(&mut self, mut unsolved_impl_items: Vec<(hir::ItemId, LocalDefId)>) {
        let mut ready;
        (ready, unsolved_impl_items) =
            unsolved_impl_items.into_iter().partition(|&(impl_id, impl_item_id)| {
                self.impl_item_with_used_self(impl_id, impl_item_id)
            });

        while !ready.is_empty() {
            self.worklist =
                ready.into_iter().map(|(_, id)| (id, ComesFromAllowExpect::No)).collect();
            self.mark_live_symbols();

            (ready, unsolved_impl_items) =
                unsolved_impl_items.into_iter().partition(|&(impl_id, impl_item_id)| {
                    self.impl_item_with_used_self(impl_id, impl_item_id)
                });
        }
    }

    fn impl_item_with_used_self(&mut self, impl_id: hir::ItemId, impl_item_id: LocalDefId) -> bool {
        if let TyKind::Path(hir::QPath::Resolved(_, path)) =
            self.tcx.hir_item(impl_id).expect_impl().self_ty.kind
            && let Res::Def(def_kind, def_id) = path.res
            && let Some(local_def_id) = def_id.as_local()
            && matches!(def_kind, DefKind::Struct | DefKind::Enum | DefKind::Union)
        {
            if self.tcx.visibility(impl_item_id).is_public() {
                // for the public method, we don't know the trait item is used or not,
                // so we mark the method live if the self is used
                return self.live_symbols.contains(&local_def_id);
            }

            if let Some(trait_item_id) = self.tcx.associated_item(impl_item_id).trait_item_def_id
                && let Some(local_id) = trait_item_id.as_local()
            {
                // for the private method, we can know the trait item is used or not,
                // so we mark the method live if the self is used and the trait item is used
                return self.live_symbols.contains(&local_id)
                    && self.live_symbols.contains(&local_def_id);
            }
        }
        false
    }
}

impl<'tcx> Visitor<'tcx> for MarkSymbolVisitor<'tcx> {
    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_maybe_typeck_results =
            self.maybe_typeck_results.replace(self.tcx.typeck_body(body));
        let body = self.tcx.hir_body(body);
        self.visit_body(body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_variant_data(&mut self, def: &'tcx hir::VariantData<'tcx>) {
        let tcx = self.tcx;
        let unconditionally_treat_fields_as_live = self.repr_unconditionally_treats_fields_as_live;
        let has_repr_simd = self.repr_has_repr_simd;
        let effective_visibilities = &tcx.effective_visibilities(());
        let live_fields = def.fields().iter().filter_map(|f| {
            let def_id = f.def_id;
            if unconditionally_treat_fields_as_live || (f.is_positional() && has_repr_simd) {
                return Some(def_id);
            }
            if !effective_visibilities.is_reachable(f.hir_id.owner.def_id) {
                return None;
            }
            if effective_visibilities.is_reachable(def_id) { Some(def_id) } else { None }
        });
        self.live_symbols.extend(live_fields);

        intravisit::walk_struct_def(self, def);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        match expr.kind {
            hir::ExprKind::Path(ref qpath @ hir::QPath::TypeRelative(..)) => {
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
            _ => (),
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) {
        // Inside the body, ignore constructions of variants
        // necessary for the pattern to match. Those construction sites
        // can't be reached unless the variant is constructed elsewhere.
        let len = self.ignore_variant_stack.len();
        self.ignore_variant_stack.extend(arm.pat.necessary_variants());
        intravisit::walk_arm(self, arm);
        self.ignore_variant_stack.truncate(len);
    }

    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
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

        intravisit::walk_pat(self, pat);
        self.in_pat = false;
    }

    fn visit_pat_expr(&mut self, expr: &'tcx rustc_hir::PatExpr<'tcx>) {
        match &expr.kind {
            rustc_hir::PatExprKind::Path(qpath) => {
                let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                self.handle_res(res);
            }
            _ => {}
        }
        intravisit::walk_pat_expr(self, expr);
    }

    fn visit_path(&mut self, path: &hir::Path<'tcx>, _: hir::HirId) {
        self.handle_res(path.res);
        intravisit::walk_path(self, path);
    }

    fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
        // When inline const blocks are used in pattern position, paths
        // referenced by it should be considered as used.
        let in_pat = mem::replace(&mut self.in_pat, false);

        self.live_symbols.insert(c.def_id);
        intravisit::walk_anon_const(self, c);

        self.in_pat = in_pat;
    }

    fn visit_inline_const(&mut self, c: &'tcx hir::ConstBlock) {
        // When inline const blocks are used in pattern position, paths
        // referenced by it should be considered as used.
        let in_pat = mem::replace(&mut self.in_pat, false);

        self.live_symbols.insert(c.def_id);
        intravisit::walk_inline_const(self, c);

        self.in_pat = in_pat;
    }
}

fn has_allow_dead_code_or_lang_attr(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> Option<ComesFromAllowExpect> {
    fn has_lang_attr(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
        tcx.has_attr(def_id, sym::lang)
            // Stable attribute for #[lang = "panic_impl"]
            || tcx.has_attr(def_id, sym::panic_handler)
    }

    fn has_allow_expect_dead_code(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
        let hir_id = tcx.local_def_id_to_hir_id(def_id);
        let lint_level = tcx.lint_level_at_node(lint::builtin::DEAD_CODE, hir_id).level;
        matches!(lint_level, lint::Allow | lint::Expect)
    }

    fn has_used_like_attr(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
        tcx.def_kind(def_id).has_codegen_attrs() && {
            let cg_attrs = tcx.codegen_fn_attrs(def_id);

            // #[used], #[no_mangle], #[export_name], etc also keeps the item alive
            // forcefully, e.g., for placing it in a specific section.
            cg_attrs.contains_extern_indicator()
                || cg_attrs.flags.contains(CodegenFnAttrFlags::USED)
                || cg_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
        }
    }

    if has_allow_expect_dead_code(tcx, def_id) {
        Some(ComesFromAllowExpect::Yes)
    } else if has_used_like_attr(tcx, def_id) || has_lang_attr(tcx, def_id) {
        Some(ComesFromAllowExpect::No)
    } else {
        None
    }
}

// These check_* functions seeds items that
//   1) We want to explicitly consider as live:
//     * Item annotated with #[allow(dead_code)]
//         - This is done so that if we want to suppress warnings for a
//           group of dead functions, we only have to annotate the "root".
//           For example, if both `f` and `g` are dead and `f` calls `g`,
//           then annotating `f` with `#[allow(dead_code)]` will suppress
//           warning for both `f` and `g`.
//     * Item annotated with #[lang=".."]
//         - This is because lang items are always callable from elsewhere.
//   or
//   2) We are not sure to be live or not
//     * Implementations of traits and trait methods
fn check_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    worklist: &mut Vec<(LocalDefId, ComesFromAllowExpect)>,
    struct_constructors: &mut LocalDefIdMap<LocalDefId>,
    unsolved_impl_items: &mut Vec<(hir::ItemId, LocalDefId)>,
    id: hir::ItemId,
) {
    let allow_dead_code = has_allow_dead_code_or_lang_attr(tcx, id.owner_id.def_id);
    if let Some(comes_from_allow) = allow_dead_code {
        worklist.push((id.owner_id.def_id, comes_from_allow));
    }

    match tcx.def_kind(id.owner_id) {
        DefKind::Enum => {
            let item = tcx.hir_item(id);
            if let hir::ItemKind::Enum(_, ref enum_def, _) = item.kind {
                if let Some(comes_from_allow) = allow_dead_code {
                    worklist.extend(
                        enum_def.variants.iter().map(|variant| (variant.def_id, comes_from_allow)),
                    );
                }

                for variant in enum_def.variants {
                    if let Some(ctor_def_id) = variant.data.ctor_def_id() {
                        struct_constructors.insert(ctor_def_id, variant.def_id);
                    }
                }
            }
        }
        DefKind::Impl { of_trait } => {
            // get DefIds from another query
            let local_def_ids = tcx
                .associated_item_def_ids(id.owner_id)
                .iter()
                .filter_map(|def_id| def_id.as_local());

            let ty_is_pub = ty_ref_to_pub_struct(tcx, tcx.hir_item(id).expect_impl().self_ty);

            // And we access the Map here to get HirId from LocalDefId
            for local_def_id in local_def_ids {
                // check the function may construct Self
                let mut may_construct_self = false;
                if let Some(fn_sig) =
                    tcx.hir_fn_sig_by_hir_id(tcx.local_def_id_to_hir_id(local_def_id))
                {
                    may_construct_self =
                        matches!(fn_sig.decl.implicit_self, hir::ImplicitSelfKind::None);
                }

                // for trait impl blocks,
                // mark the method live if the self_ty is public,
                // or the method is public and may construct self
                if of_trait
                    && (!matches!(tcx.def_kind(local_def_id), DefKind::AssocFn)
                        || tcx.visibility(local_def_id).is_public()
                            && (ty_is_pub || may_construct_self))
                {
                    worklist.push((local_def_id, ComesFromAllowExpect::No));
                } else if let Some(comes_from_allow) =
                    has_allow_dead_code_or_lang_attr(tcx, local_def_id)
                {
                    worklist.push((local_def_id, comes_from_allow));
                } else if of_trait {
                    // private method || public method not constructs self
                    unsolved_impl_items.push((id, local_def_id));
                }
            }
        }
        DefKind::Struct => {
            let item = tcx.hir_item(id);
            if let hir::ItemKind::Struct(_, ref variant_data, _) = item.kind
                && let Some(ctor_def_id) = variant_data.ctor_def_id()
            {
                struct_constructors.insert(ctor_def_id, item.owner_id.def_id);
            }
        }
        DefKind::GlobalAsm => {
            // global_asm! is always live.
            worklist.push((id.owner_id.def_id, ComesFromAllowExpect::No));
        }
        _ => {}
    }
}

fn check_trait_item(
    tcx: TyCtxt<'_>,
    worklist: &mut Vec<(LocalDefId, ComesFromAllowExpect)>,
    id: hir::TraitItemId,
) {
    use hir::TraitItemKind::{Const, Fn};
    if matches!(tcx.def_kind(id.owner_id), DefKind::AssocConst | DefKind::AssocFn) {
        let trait_item = tcx.hir_trait_item(id);
        if matches!(trait_item.kind, Const(_, Some(_)) | Fn(..))
            && let Some(comes_from_allow) =
                has_allow_dead_code_or_lang_attr(tcx, trait_item.owner_id.def_id)
        {
            worklist.push((trait_item.owner_id.def_id, comes_from_allow));
        }
    }
}

fn check_foreign_item(
    tcx: TyCtxt<'_>,
    worklist: &mut Vec<(LocalDefId, ComesFromAllowExpect)>,
    id: hir::ForeignItemId,
) {
    if matches!(tcx.def_kind(id.owner_id), DefKind::Static { .. } | DefKind::Fn)
        && let Some(comes_from_allow) = has_allow_dead_code_or_lang_attr(tcx, id.owner_id.def_id)
    {
        worklist.push((id.owner_id.def_id, comes_from_allow));
    }
}

fn create_and_seed_worklist(
    tcx: TyCtxt<'_>,
) -> (
    Vec<(LocalDefId, ComesFromAllowExpect)>,
    LocalDefIdMap<LocalDefId>,
    Vec<(hir::ItemId, LocalDefId)>,
) {
    let effective_visibilities = &tcx.effective_visibilities(());
    // see `MarkSymbolVisitor::struct_constructors`
    let mut unsolved_impl_item = Vec::new();
    let mut struct_constructors = Default::default();
    let mut worklist = effective_visibilities
        .iter()
        .filter_map(|(&id, effective_vis)| {
            effective_vis
                .is_public_at_level(Level::Reachable)
                .then_some(id)
                .map(|id| (id, ComesFromAllowExpect::No))
        })
        // Seed entry point
        .chain(
            tcx.entry_fn(())
                .and_then(|(def_id, _)| def_id.as_local().map(|id| (id, ComesFromAllowExpect::No))),
        )
        .collect::<Vec<_>>();

    let crate_items = tcx.hir_crate_items(());
    for id in crate_items.free_items() {
        check_item(tcx, &mut worklist, &mut struct_constructors, &mut unsolved_impl_item, id);
    }

    for id in crate_items.trait_items() {
        check_trait_item(tcx, &mut worklist, id);
    }

    for id in crate_items.foreign_items() {
        check_foreign_item(tcx, &mut worklist, id);
    }

    (worklist, struct_constructors, unsolved_impl_item)
}

fn live_symbols_and_ignored_derived_traits(
    tcx: TyCtxt<'_>,
    (): (),
) -> (LocalDefIdSet, LocalDefIdMap<Vec<(DefId, DefId)>>) {
    let (worklist, struct_constructors, unsolved_impl_items) = create_and_seed_worklist(tcx);
    let mut symbol_visitor = MarkSymbolVisitor {
        worklist,
        tcx,
        maybe_typeck_results: None,
        live_symbols: Default::default(),
        repr_unconditionally_treats_fields_as_live: false,
        repr_has_repr_simd: false,
        in_pat: false,
        ignore_variant_stack: vec![],
        struct_constructors,
        ignored_derived_traits: Default::default(),
    };
    symbol_visitor.mark_live_symbols();
    symbol_visitor.solve_rest_impl_items(unsolved_impl_items);

    (symbol_visitor.live_symbols, symbol_visitor.ignored_derived_traits)
}

struct DeadItem {
    def_id: LocalDefId,
    name: Symbol,
    level: (lint::Level, Option<LintExpectationId>),
}

struct DeadVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    live_symbols: &'tcx LocalDefIdSet,
    ignored_derived_traits: &'tcx LocalDefIdMap<Vec<(DefId, DefId)>>,
}

enum ShouldWarnAboutField {
    Yes,
    No,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ReportOn {
    TupleField,
    NamedField,
}

impl<'tcx> DeadVisitor<'tcx> {
    fn should_warn_about_field(&mut self, field: &ty::FieldDef) -> ShouldWarnAboutField {
        if self.live_symbols.contains(&field.did.expect_local()) {
            return ShouldWarnAboutField::No;
        }
        let field_type = self.tcx.type_of(field.did).instantiate_identity();
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

    fn def_lint_level(&self, id: LocalDefId) -> (lint::Level, Option<LintExpectationId>) {
        let hir_id = self.tcx.local_def_id_to_hir_id(id);
        let level = self.tcx.lint_level_at_node(DEAD_CODE, hir_id);
        (level.level, level.lint_id)
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
        fn get_parent_if_enum_variant<'tcx>(
            tcx: TyCtxt<'tcx>,
            may_variant: LocalDefId,
        ) -> LocalDefId {
            if let Node::Variant(_) = tcx.hir_node_by_def_id(may_variant)
                && let Some(enum_did) = tcx.opt_parent(may_variant.to_def_id())
                && let Some(enum_local_id) = enum_did.as_local()
                && let Node::Item(item) = tcx.hir_node_by_def_id(enum_local_id)
                && let ItemKind::Enum(..) = item.kind
            {
                enum_local_id
            } else {
                may_variant
            }
        }

        let Some(&first_item) = dead_codes.first() else {
            return;
        };
        let tcx = self.tcx;

        let first_lint_level = first_item.level;
        assert!(dead_codes.iter().skip(1).all(|item| item.level == first_lint_level));

        let names: Vec<_> = dead_codes.iter().map(|item| item.name).collect();
        let spans: Vec<_> = dead_codes
            .iter()
            .map(|item| match tcx.def_ident_span(item.def_id) {
                Some(s) => s.with_ctxt(tcx.def_span(item.def_id).ctxt()),
                None => tcx.def_span(item.def_id),
            })
            .collect();

        let descr = tcx.def_descr(first_item.def_id.to_def_id());
        // `impl` blocks are "batched" and (unlike other batching) might
        // contain different kinds of associated items.
        let descr = if dead_codes.iter().any(|item| tcx.def_descr(item.def_id.to_def_id()) != descr)
        {
            "associated item"
        } else {
            descr
        };
        let num = dead_codes.len();
        let multiple = num > 6;
        let name_list = names.into();

        let parent_info = if let Some(parent_item) = parent_item {
            let parent_descr = tcx.def_descr(parent_item.to_def_id());
            let span = if let DefKind::Impl { .. } = tcx.def_kind(parent_item) {
                tcx.def_span(parent_item)
            } else {
                tcx.def_ident_span(parent_item).unwrap()
            };
            Some(ParentInfo { num, descr, parent_descr, span })
        } else {
            None
        };

        let encl_def_id = parent_item.unwrap_or(first_item.def_id);
        // If parent of encl_def_id is an enum, use the parent ID instead.
        let encl_def_id = get_parent_if_enum_variant(tcx, encl_def_id);

        let ignored_derived_impls =
            if let Some(ign_traits) = self.ignored_derived_traits.get(&encl_def_id) {
                let trait_list = ign_traits
                    .iter()
                    .map(|(trait_id, _)| self.tcx.item_name(*trait_id))
                    .collect::<Vec<_>>();
                let trait_list_len = trait_list.len();
                Some(IgnoredDerivedImpls {
                    name: self.tcx.item_name(encl_def_id.to_def_id()),
                    trait_list: trait_list.into(),
                    trait_list_len,
                })
            } else {
                None
            };

        let diag = match report_on {
            ReportOn::TupleField => {
                let tuple_fields = if let Some(parent_id) = parent_item
                    && let node = tcx.hir_node_by_def_id(parent_id)
                    && let hir::Node::Item(hir::Item {
                        kind: hir::ItemKind::Struct(_, hir::VariantData::Tuple(fields, _, _), _),
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
                    change_fields_suggestion: fields_suggestion,
                    parent_info,
                    ignored_derived_impls,
                }
            }
            ReportOn::NamedField => MultipleDeadCodes::DeadCodes {
                multiple,
                num,
                descr,
                participle,
                name_list,
                parent_info,
                ignored_derived_impls,
            },
        };

        let hir_id = tcx.local_def_id_to_hir_id(first_item.def_id);
        self.tcx.emit_node_span_lint(DEAD_CODE, hir_id, MultiSpan::from_spans(spans), diag);
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
        // FIXME: `dead_codes` should probably be morally equivalent to `IndexMap<(Level, LintExpectationId), (DefId, Symbol)>`
        dead_codes.sort_by_key(|v| v.level.0);
        for group in dead_codes.chunk_by(|a, b| a.level == b.level) {
            self.lint_at_single_level(&group, participle, Some(def_id), report_on);
        }
    }

    fn warn_dead_code(&mut self, id: LocalDefId, participle: &str) {
        let item = DeadItem {
            def_id: id,
            name: self.tcx.item_name(id.to_def_id()),
            level: self.def_lint_level(id),
        };
        self.lint_at_single_level(&[&item], participle, None, ReportOn::NamedField);
    }

    fn check_definition(&mut self, def_id: LocalDefId) {
        if self.is_live_code(def_id) {
            return;
        }
        match self.tcx.def_kind(def_id) {
            DefKind::AssocConst
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

    fn is_live_code(&self, def_id: LocalDefId) -> bool {
        // if we cannot get a name for the item, then we just assume that it is
        // live. I mean, we can't really emit a lint.
        let Some(name) = self.tcx.opt_item_name(def_id.to_def_id()) else {
            return true;
        };

        self.live_symbols.contains(&def_id) || name.as_str().starts_with('_')
    }
}

fn check_mod_deathness(tcx: TyCtxt<'_>, module: LocalModDefId) {
    let (live_symbols, ignored_derived_traits) = tcx.live_symbols_and_ignored_derived_traits(());
    let mut visitor = DeadVisitor { tcx, live_symbols, ignored_derived_traits };

    let module_items = tcx.hir_module_items(module);

    for item in module_items.free_items() {
        let def_kind = tcx.def_kind(item.owner_id);

        let mut dead_codes = Vec::new();
        // if we have diagnosed the trait, do not diagnose unused methods
        if matches!(def_kind, DefKind::Impl { .. })
            || (def_kind == DefKind::Trait && live_symbols.contains(&item.owner_id.def_id))
        {
            for &def_id in tcx.associated_item_def_ids(item.owner_id.def_id) {
                // We have diagnosed unused methods in traits
                if matches!(def_kind, DefKind::Impl { of_trait: true })
                    && tcx.def_kind(def_id) == DefKind::AssocFn
                    || def_kind == DefKind::Trait && tcx.def_kind(def_id) != DefKind::AssocFn
                {
                    continue;
                }

                if let Some(local_def_id) = def_id.as_local()
                    && !visitor.is_live_code(local_def_id)
                {
                    let name = tcx.item_name(def_id);
                    let level = visitor.def_lint_level(local_def_id);
                    dead_codes.push(DeadItem { def_id: local_def_id, name, level });
                }
            }
        }
        if !dead_codes.is_empty() {
            visitor.warn_multiple(item.owner_id.def_id, "used", dead_codes, ReportOn::NamedField);
        }

        if !live_symbols.contains(&item.owner_id.def_id) {
            let parent = tcx.local_parent(item.owner_id.def_id);
            if parent != module.to_local_def_id() && !live_symbols.contains(&parent) {
                // We already have diagnosed something.
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
                if !live_symbols.contains(&def_id) {
                    // Record to group diagnostics.
                    let level = visitor.def_lint_level(def_id);
                    dead_variants.push(DeadItem { def_id, name: variant.name, level });
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
                            let level = visitor.def_lint_level(def_id);
                            Some(DeadItem { def_id, name: field.name, level })
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

    for foreign_item in module_items.foreign_items() {
        visitor.check_definition(foreign_item.owner_id.def_id);
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers =
        Providers { live_symbols_and_ignored_derived_traits, check_mod_deathness, ..*providers };
}
