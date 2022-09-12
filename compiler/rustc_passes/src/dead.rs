// This implements the dead-code warning pass. It follows middle::reachable
// closely. The idea is that all reachable symbols are live, codes called
// from live codes are live, and everything else is dead.

use itertools::Itertools;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, Applicability, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Node, PatKind, TyKind};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::privacy;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, DefIdTree, TyCtxt};
use rustc_session::lint;
use rustc_span::symbol::{sym, Symbol};
use std::mem;

// Any local node that may call something in its body block should be
// explored. For example, if it's a live Node::Item that is a
// function, then we should explore its block to check for codes that
// may need to be marked as live.
fn should_explore(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    matches!(
        tcx.hir().find_by_def_id(def_id),
        Some(
            Node::Item(..)
                | Node::ImplItem(..)
                | Node::ForeignItem(..)
                | Node::TraitItem(..)
                | Node::Variant(..)
                | Node::AnonConst(..)
        )
    )
}

struct MarkSymbolVisitor<'tcx> {
    worklist: Vec<LocalDefId>,
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Option<&'tcx ty::TypeckResults<'tcx>>,
    live_symbols: FxHashSet<LocalDefId>,
    repr_has_repr_c: bool,
    repr_has_repr_simd: bool,
    in_pat: bool,
    ignore_variant_stack: Vec<DefId>,
    // maps from tuple struct constructors to tuple struct items
    struct_constructors: FxHashMap<LocalDefId, LocalDefId>,
    // maps from ADTs to ignored derived traits (e.g. Debug and Clone)
    // and the span of their respective impl (i.e., part of the derive
    // macro)
    ignored_derived_traits: FxHashMap<LocalDefId, Vec<(DefId, DefId)>>,
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
                self.worklist.push(def_id);
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
            Res::SelfTy { trait_: t, alias_to: i } => {
                if let Some(t) = t {
                    self.check_def_id(t);
                }
                if let Some((i, _)) = i {
                    self.check_def_id(i);
                }
            }
            Res::ToolMod | Res::NonMacroAttr(..) | Res::Err => {}
        }
    }

    fn lookup_and_handle_method(&mut self, id: hir::HirId) {
        if let Some(def_id) = self.typeck_results().type_dependent_def_id(id) {
            self.check_def_id(def_id);
        } else {
            bug!("no type-dependent def for method");
        }
    }

    fn handle_field_access(&mut self, lhs: &hir::Expr<'_>, hir_id: hir::HirId) {
        match self.typeck_results().expr_ty_adjusted(lhs).kind() {
            ty::Adt(def, _) => {
                let index = self.tcx.field_index(hir_id, self.typeck_results());
                self.insert_def_id(def.non_enum_variant().fields[index].did);
            }
            ty::Tuple(..) => {}
            _ => span_bug!(lhs.span, "named field access on non-ADT"),
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
                (hir::ExprKind::Path(ref qpath_l), hir::ExprKind::Path(ref qpath_r)) => {
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
                self.tcx.struct_span_lint_hir(
                    lint::builtin::DEAD_CODE,
                    assign.hir_id,
                    assign.span,
                    |lint| {
                        lint.build(&format!(
                            "useless assignment of {} of type `{}` to itself",
                            if is_field_assign { "field" } else { "variable" },
                            self.typeck_results().expr_ty(lhs),
                        ))
                        .emit();
                    },
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
            let index = self.tcx.field_index(pat.hir_id, self.typeck_results());
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
            _ => span_bug!(lhs.span, "non-ADT in tuple struct pattern"),
        };
        let dotdot = dotdot.as_opt_usize().unwrap_or(pats.len());
        let first_n = pats.iter().enumerate().take(dotdot);
        let missing = variant.fields.len() - pats.len();
        let last_n = pats.iter().enumerate().skip(dotdot).map(|(idx, pat)| (idx + missing, pat));
        for (idx, pat) in first_n.chain(last_n) {
            if let PatKind::Wild = pat.kind {
                continue;
            }
            self.insert_def_id(variant.fields[idx].did);
        }
    }

    fn mark_live_symbols(&mut self) {
        let mut scanned = FxHashSet::default();
        while let Some(id) = self.worklist.pop() {
            if !scanned.insert(id) {
                continue;
            }

            // in the case of tuple struct constructors we want to check the item, not the generated
            // tuple struct constructor function
            let id = self.struct_constructors.get(&id).copied().unwrap_or(id);

            if let Some(node) = self.tcx.hir().find_by_def_id(id) {
                self.live_symbols.insert(id);
                self.visit_node(node);
            }
        }
    }

    /// Automatically generated items marked with `rustc_trivial_field_reads`
    /// will be ignored for the purposes of dead code analysis (see PR #85200
    /// for discussion).
    fn should_ignore_item(&mut self, def_id: DefId) -> bool {
        if let Some(impl_of) = self.tcx.impl_of_method(def_id) {
            if !self.tcx.has_attr(impl_of, sym::automatically_derived) {
                return false;
            }

            if let Some(trait_of) = self.tcx.trait_id_of_impl(impl_of)
                && self.tcx.has_attr(trait_of, sym::rustc_trivial_field_reads)
            {
                let trait_ref = self.tcx.impl_trait_ref(impl_of).unwrap();
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

        return false;
    }

    fn visit_node(&mut self, node: Node<'tcx>) {
        if let Node::ImplItem(hir::ImplItem { def_id, .. }) = node
            && self.should_ignore_item(def_id.to_def_id())
        {
            return;
        }

        let had_repr_c = self.repr_has_repr_c;
        let had_repr_simd = self.repr_has_repr_simd;
        self.repr_has_repr_c = false;
        self.repr_has_repr_simd = false;
        match node {
            Node::Item(item) => match item.kind {
                hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) => {
                    let def = self.tcx.adt_def(item.def_id);
                    self.repr_has_repr_c = def.repr().c();
                    self.repr_has_repr_simd = def.repr().simd();

                    intravisit::walk_item(self, &item)
                }
                hir::ItemKind::ForeignMod { .. } => {}
                _ => intravisit::walk_item(self, &item),
            },
            Node::TraitItem(trait_item) => {
                intravisit::walk_trait_item(self, trait_item);
            }
            Node::ImplItem(impl_item) => {
                let item = self.tcx.local_parent(impl_item.def_id);
                if self.tcx.impl_trait_ref(item).is_none() {
                    //// If it's a type whose items are live, then it's live, too.
                    //// This is done to handle the case where, for example, the static
                    //// method of a private type is used, but the type itself is never
                    //// called directly.
                    let self_ty = self.tcx.type_of(item);
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
                intravisit::walk_foreign_item(self, &foreign_item);
            }
            _ => {}
        }
        self.repr_has_repr_simd = had_repr_simd;
        self.repr_has_repr_c = had_repr_c;
    }

    fn mark_as_used_if_union(&mut self, adt: ty::AdtDef<'tcx>, fields: &[hir::ExprField<'_>]) {
        if adt.is_union() && adt.non_enum_variant().fields.len() > 1 && adt.did().is_local() {
            for field in fields {
                let index = self.tcx.field_index(field.hir_id, self.typeck_results());
                self.insert_def_id(adt.non_enum_variant().fields[index].did);
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for MarkSymbolVisitor<'tcx> {
    fn visit_nested_body(&mut self, body: hir::BodyId) {
        let old_maybe_typeck_results =
            self.maybe_typeck_results.replace(self.tcx.typeck_body(body));
        let body = self.tcx.hir().body(body);
        self.visit_body(body);
        self.maybe_typeck_results = old_maybe_typeck_results;
    }

    fn visit_variant_data(&mut self, def: &'tcx hir::VariantData<'tcx>) {
        let tcx = self.tcx;
        let has_repr_c = self.repr_has_repr_c;
        let has_repr_simd = self.repr_has_repr_simd;
        let live_fields = def.fields().iter().filter_map(|f| {
            let def_id = tcx.hir().local_def_id(f.hir_id);
            if has_repr_c || (f.is_positional() && has_repr_simd) {
                return Some(def_id);
            }
            if !tcx.visibility(f.hir_id.owner).is_public() {
                return None;
            }
            if tcx.visibility(def_id).is_public() { Some(def_id) } else { None }
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
                self.handle_field_access(&lhs, expr.hir_id);
            }
            hir::ExprKind::Struct(ref qpath, ref fields, _) => {
                let res = self.typeck_results().qpath_res(qpath, expr.hir_id);
                self.handle_res(res);
                if let ty::Adt(adt, _) = self.typeck_results().expr_ty(expr).kind() {
                    self.mark_as_used_if_union(*adt, fields);
                }
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
            PatKind::Struct(ref path, ref fields, _) => {
                let res = self.typeck_results().qpath_res(path, pat.hir_id);
                self.handle_field_pattern_match(pat, res, fields);
            }
            PatKind::Path(ref qpath) => {
                let res = self.typeck_results().qpath_res(qpath, pat.hir_id);
                self.handle_res(res);
            }
            PatKind::TupleStruct(ref qpath, ref fields, dotdot) => {
                let res = self.typeck_results().qpath_res(qpath, pat.hir_id);
                self.handle_tuple_field_pattern_match(pat, res, fields, dotdot);
            }
            _ => (),
        }

        intravisit::walk_pat(self, pat);
        self.in_pat = false;
    }

    fn visit_path(&mut self, path: &'tcx hir::Path<'tcx>, _: hir::HirId) {
        self.handle_res(path.res);
        intravisit::walk_path(self, path);
    }

    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
        if let TyKind::OpaqueDef(item_id, _, _) = ty.kind {
            let item = self.tcx.hir().item(item_id);
            intravisit::walk_item(self, item);
        }
        intravisit::walk_ty(self, ty);
    }

    fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
        // When inline const blocks are used in pattern position, paths
        // referenced by it should be considered as used.
        let in_pat = mem::replace(&mut self.in_pat, false);

        self.live_symbols.insert(self.tcx.hir().local_def_id(c.hir_id));
        intravisit::walk_anon_const(self, c);

        self.in_pat = in_pat;
    }
}

fn has_allow_dead_code_or_lang_attr_helper(
    tcx: TyCtxt<'_>,
    id: hir::HirId,
    lint: &'static lint::Lint,
) -> bool {
    let attrs = tcx.hir().attrs(id);
    if tcx.sess.contains_name(attrs, sym::lang) {
        return true;
    }

    // Stable attribute for #[lang = "panic_impl"]
    if tcx.sess.contains_name(attrs, sym::panic_handler) {
        return true;
    }

    // (To be) stable attribute for #[lang = "oom"]
    if tcx.sess.contains_name(attrs, sym::alloc_error_handler) {
        return true;
    }

    let def_id = tcx.hir().local_def_id(id);
    if tcx.def_kind(def_id).has_codegen_attrs() {
        let cg_attrs = tcx.codegen_fn_attrs(def_id);

        // #[used], #[no_mangle], #[export_name], etc also keeps the item alive
        // forcefully, e.g., for placing it in a specific section.
        if cg_attrs.contains_extern_indicator()
            || cg_attrs.flags.contains(CodegenFnAttrFlags::USED)
            || cg_attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
        {
            return true;
        }
    }

    tcx.lint_level_at_node(lint, id).0 == lint::Allow
}

fn has_allow_dead_code_or_lang_attr(tcx: TyCtxt<'_>, id: hir::HirId) -> bool {
    has_allow_dead_code_or_lang_attr_helper(tcx, id, lint::builtin::DEAD_CODE)
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
    worklist: &mut Vec<LocalDefId>,
    struct_constructors: &mut FxHashMap<LocalDefId, LocalDefId>,
    id: hir::ItemId,
) {
    let allow_dead_code = has_allow_dead_code_or_lang_attr(tcx, id.hir_id());
    if allow_dead_code {
        worklist.push(id.def_id);
    }

    match tcx.def_kind(id.def_id) {
        DefKind::Enum => {
            let item = tcx.hir().item(id);
            if let hir::ItemKind::Enum(ref enum_def, _) = item.kind {
                let hir = tcx.hir();
                if allow_dead_code {
                    worklist.extend(
                        enum_def.variants.iter().map(|variant| hir.local_def_id(variant.id)),
                    );
                }

                for variant in enum_def.variants {
                    if let Some(ctor_hir_id) = variant.data.ctor_hir_id() {
                        struct_constructors
                            .insert(hir.local_def_id(ctor_hir_id), hir.local_def_id(variant.id));
                    }
                }
            }
        }
        DefKind::Impl => {
            let of_trait = tcx.impl_trait_ref(id.def_id);

            if of_trait.is_some() {
                worklist.push(id.def_id);
            }

            // get DefIds from another query
            let local_def_ids = tcx
                .associated_item_def_ids(id.def_id)
                .iter()
                .filter_map(|def_id| def_id.as_local());

            // And we access the Map here to get HirId from LocalDefId
            for id in local_def_ids {
                if of_trait.is_some()
                    || has_allow_dead_code_or_lang_attr(tcx, tcx.hir().local_def_id_to_hir_id(id))
                {
                    worklist.push(id);
                }
            }
        }
        DefKind::Struct => {
            let item = tcx.hir().item(id);
            if let hir::ItemKind::Struct(ref variant_data, _) = item.kind
                && let Some(ctor_hir_id) = variant_data.ctor_hir_id()
            {
                struct_constructors.insert(tcx.hir().local_def_id(ctor_hir_id), item.def_id);
            }
        }
        DefKind::GlobalAsm => {
            // global_asm! is always live.
            worklist.push(id.def_id);
        }
        _ => {}
    }
}

fn check_trait_item<'tcx>(tcx: TyCtxt<'tcx>, worklist: &mut Vec<LocalDefId>, id: hir::TraitItemId) {
    use hir::TraitItemKind::{Const, Fn};
    if matches!(tcx.def_kind(id.def_id), DefKind::AssocConst | DefKind::AssocFn) {
        let trait_item = tcx.hir().trait_item(id);
        if matches!(trait_item.kind, Const(_, Some(_)) | Fn(_, hir::TraitFn::Provided(_)))
            && has_allow_dead_code_or_lang_attr(tcx, trait_item.hir_id())
        {
            worklist.push(trait_item.def_id);
        }
    }
}

fn check_foreign_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    worklist: &mut Vec<LocalDefId>,
    id: hir::ForeignItemId,
) {
    if matches!(tcx.def_kind(id.def_id), DefKind::Static(_) | DefKind::Fn)
        && has_allow_dead_code_or_lang_attr(tcx, id.hir_id())
    {
        worklist.push(id.def_id);
    }
}

fn create_and_seed_worklist<'tcx>(
    tcx: TyCtxt<'tcx>,
) -> (Vec<LocalDefId>, FxHashMap<LocalDefId, LocalDefId>) {
    let access_levels = &tcx.privacy_access_levels(());
    // see `MarkSymbolVisitor::struct_constructors`
    let mut struct_constructors = Default::default();
    let mut worklist = access_levels
        .map
        .iter()
        .filter_map(
            |(&id, &level)| {
                if level >= privacy::AccessLevel::Reachable { Some(id) } else { None }
            },
        )
        // Seed entry point
        .chain(tcx.entry_fn(()).and_then(|(def_id, _)| def_id.as_local()))
        .collect::<Vec<_>>();

    let crate_items = tcx.hir_crate_items(());
    for id in crate_items.items() {
        check_item(tcx, &mut worklist, &mut struct_constructors, id);
    }

    for id in crate_items.trait_items() {
        check_trait_item(tcx, &mut worklist, id);
    }

    for id in crate_items.foreign_items() {
        check_foreign_item(tcx, &mut worklist, id);
    }

    (worklist, struct_constructors)
}

fn live_symbols_and_ignored_derived_traits<'tcx>(
    tcx: TyCtxt<'tcx>,
    (): (),
) -> (FxHashSet<LocalDefId>, FxHashMap<LocalDefId, Vec<(DefId, DefId)>>) {
    let (worklist, struct_constructors) = create_and_seed_worklist(tcx);
    let mut symbol_visitor = MarkSymbolVisitor {
        worklist,
        tcx,
        maybe_typeck_results: None,
        live_symbols: Default::default(),
        repr_has_repr_c: false,
        repr_has_repr_simd: false,
        in_pat: false,
        ignore_variant_stack: vec![],
        struct_constructors,
        ignored_derived_traits: FxHashMap::default(),
    };
    symbol_visitor.mark_live_symbols();
    (symbol_visitor.live_symbols, symbol_visitor.ignored_derived_traits)
}

struct DeadVariant {
    def_id: LocalDefId,
    name: Symbol,
    level: lint::Level,
}

struct DeadVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    live_symbols: &'tcx FxHashSet<LocalDefId>,
    ignored_derived_traits: &'tcx FxHashMap<LocalDefId, Vec<(DefId, DefId)>>,
}

enum ShouldWarnAboutField {
    Yes(bool), // positional?
    No,
}

impl<'tcx> DeadVisitor<'tcx> {
    fn should_warn_about_field(&mut self, field: &ty::FieldDef) -> ShouldWarnAboutField {
        if self.live_symbols.contains(&field.did.expect_local()) {
            return ShouldWarnAboutField::No;
        }
        let field_type = self.tcx.type_of(field.did);
        if field_type.is_phantom_data() {
            return ShouldWarnAboutField::No;
        }
        let is_positional = field.name.as_str().starts_with(|c: char| c.is_ascii_digit());
        if is_positional
            && self
                .tcx
                .layout_of(self.tcx.param_env(field.did).and(field_type))
                .map_or(true, |layout| layout.is_zst())
        {
            return ShouldWarnAboutField::No;
        }
        ShouldWarnAboutField::Yes(is_positional)
    }

    fn warn_multiple_dead_codes(
        &self,
        dead_codes: &[LocalDefId],
        participle: &str,
        parent_item: Option<LocalDefId>,
        is_positional: bool,
    ) {
        if let Some(&first_id) = dead_codes.first() {
            let tcx = self.tcx;
            let names: Vec<_> = dead_codes
                .iter()
                .map(|&def_id| tcx.item_name(def_id.to_def_id()).to_string())
                .collect();
            let spans: Vec<_> = dead_codes
                .iter()
                .map(|&def_id| match tcx.def_ident_span(def_id) {
                    Some(s) => s.with_ctxt(tcx.def_span(def_id).ctxt()),
                    None => tcx.def_span(def_id),
                })
                .collect();

            tcx.struct_span_lint_hir(
                if is_positional {
                    lint::builtin::UNUSED_TUPLE_STRUCT_FIELDS
                } else {
                    lint::builtin::DEAD_CODE
                },
                tcx.hir().local_def_id_to_hir_id(first_id),
                MultiSpan::from_spans(spans.clone()),
                |lint| {
                    let descr = tcx.def_kind(first_id).descr(first_id.to_def_id());
                    let span_len = dead_codes.len();
                    let names = match &names[..] {
                        _ if span_len > 6 => String::new(),
                        [name] => format!("`{name}` "),
                        [names @ .., last] => {
                            format!(
                                "{} and `{last}` ",
                                names.iter().map(|name| format!("`{name}`")).join(", ")
                            )
                        }
                        [] => unreachable!(),
                    };
                    let mut err = lint.build(&format!(
                        "{these}{descr}{s} {names}{are} never {participle}",
                        these = if span_len > 6 { "multiple " } else { "" },
                        s = pluralize!(span_len),
                        are = pluralize!("is", span_len),
                    ));

                    if is_positional {
                        err.multipart_suggestion(
                            &format!(
                                "consider changing the field{s} to be of unit type to \
                                      suppress this warning while preserving the field \
                                      numbering, or remove the field{s}",
                                s = pluralize!(span_len)
                            ),
                            spans.iter().map(|sp| (*sp, "()".to_string())).collect(),
                            // "HasPlaceholders" because applying this fix by itself isn't
                            // enough: All constructor calls have to be adjusted as well
                            Applicability::HasPlaceholders,
                        );
                    }

                    if let Some(parent_item) = parent_item {
                        let parent_descr = tcx.def_kind(parent_item).descr(parent_item.to_def_id());
                        err.span_label(
                            tcx.def_ident_span(parent_item).unwrap(),
                            format!("{descr}{s} in this {parent_descr}", s = pluralize!(span_len)),
                        );
                    }

                    let encl_def_id = parent_item.unwrap_or(first_id);
                    if let Some(ign_traits) = self.ignored_derived_traits.get(&encl_def_id) {
                        let traits_str = ign_traits
                            .iter()
                            .map(|(trait_id, _)| format!("`{}`", self.tcx.item_name(*trait_id)))
                            .collect::<Vec<_>>()
                            .join(" and ");
                        let plural_s = pluralize!(ign_traits.len());
                        let article = if ign_traits.len() > 1 { "" } else { "a " };
                        let is_are = if ign_traits.len() > 1 { "these are" } else { "this is" };
                        let msg = format!(
                            "`{}` has {}derived impl{} for the trait{} {}, but {} \
                            intentionally ignored during dead code analysis",
                            self.tcx.item_name(encl_def_id.to_def_id()),
                            article,
                            plural_s,
                            plural_s,
                            traits_str,
                            is_are
                        );
                        err.note(&msg);
                    }
                    err.emit();
                },
            );
        }
    }

    fn warn_dead_fields_and_variants(
        &self,
        def_id: LocalDefId,
        participle: &str,
        dead_codes: Vec<DeadVariant>,
        is_positional: bool,
    ) {
        let mut dead_codes = dead_codes
            .iter()
            .filter(|v| !v.name.as_str().starts_with('_'))
            .map(|v| v)
            .collect::<Vec<&DeadVariant>>();
        if dead_codes.is_empty() {
            return;
        }
        dead_codes.sort_by_key(|v| v.level);
        for (_, group) in &dead_codes.into_iter().group_by(|v| v.level) {
            self.warn_multiple_dead_codes(
                &group.map(|v| v.def_id).collect::<Vec<_>>(),
                participle,
                Some(def_id),
                is_positional,
            );
        }
    }

    fn warn_dead_code(&mut self, id: LocalDefId, participle: &str) {
        self.warn_multiple_dead_codes(&[id], participle, None, false);
    }

    fn check_definition(&mut self, def_id: LocalDefId) {
        if self.live_symbols.contains(&def_id) {
            return;
        }
        let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
        if has_allow_dead_code_or_lang_attr(self.tcx, hir_id) {
            return;
        }
        let Some(name) = self.tcx.opt_item_name(def_id.to_def_id()) else {
            return
        };
        if name.as_str().starts_with('_') {
            return;
        }
        match self.tcx.def_kind(def_id) {
            DefKind::AssocConst
            | DefKind::AssocFn
            | DefKind::Fn
            | DefKind::Static(_)
            | DefKind::Const
            | DefKind::TyAlias
            | DefKind::Enum
            | DefKind::Union
            | DefKind::ForeignTy => self.warn_dead_code(def_id, "used"),
            DefKind::Struct => self.warn_dead_code(def_id, "constructed"),
            DefKind::Variant | DefKind::Field => bug!("should be handled specially"),
            _ => {}
        }
    }
}

fn check_mod_deathness(tcx: TyCtxt<'_>, module: LocalDefId) {
    let (live_symbols, ignored_derived_traits) = tcx.live_symbols_and_ignored_derived_traits(());
    let mut visitor = DeadVisitor { tcx, live_symbols, ignored_derived_traits };

    let module_items = tcx.hir_module_items(module);

    for item in module_items.items() {
        if !live_symbols.contains(&item.def_id) {
            let parent = tcx.local_parent(item.def_id);
            if parent != module && !live_symbols.contains(&parent) {
                // We already have diagnosed something.
                continue;
            }
            visitor.check_definition(item.def_id);
            continue;
        }

        let def_kind = tcx.def_kind(item.def_id);
        if let DefKind::Struct | DefKind::Union | DefKind::Enum = def_kind {
            let adt = tcx.adt_def(item.def_id);
            let mut dead_variants = Vec::new();

            for variant in adt.variants() {
                let def_id = variant.def_id.expect_local();
                if !live_symbols.contains(&def_id) {
                    // Record to group diagnostics.
                    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
                    let level = tcx.lint_level_at_node(lint::builtin::DEAD_CODE, hir_id).0;
                    dead_variants.push(DeadVariant { def_id, name: variant.name, level });
                    continue;
                }

                let mut is_positional = false;
                let dead_fields = variant
                    .fields
                    .iter()
                    .filter_map(|field| {
                        let def_id = field.did.expect_local();
                        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
                        if let ShouldWarnAboutField::Yes(is_pos) =
                            visitor.should_warn_about_field(&field)
                        {
                            let level = tcx
                                .lint_level_at_node(
                                    if is_pos {
                                        is_positional = true;
                                        lint::builtin::UNUSED_TUPLE_STRUCT_FIELDS
                                    } else {
                                        lint::builtin::DEAD_CODE
                                    },
                                    hir_id,
                                )
                                .0;
                            Some(DeadVariant { def_id, name: field.name, level })
                        } else {
                            None
                        }
                    })
                    .collect();
                visitor.warn_dead_fields_and_variants(def_id, "read", dead_fields, is_positional)
            }

            visitor.warn_dead_fields_and_variants(item.def_id, "constructed", dead_variants, false);
        }
    }

    for impl_item in module_items.impl_items() {
        visitor.check_definition(impl_item.def_id);
    }

    for foreign_item in module_items.foreign_items() {
        visitor.check_definition(foreign_item.def_id);
    }

    // We do not warn trait items.
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers =
        Providers { live_symbols_and_ignored_derived_traits, check_mod_deathness, ..*providers };
}
