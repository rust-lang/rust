// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.

use crate::check::FnCtxt;

use rustc::hir::map::Map;
use rustc::ty::adjustment::{Adjust, Adjustment, PointerCast};
use rustc::ty::fold::{TypeFoldable, TypeFolder};
use rustc::ty::{self, Ty, TyCtxt};
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, DefIdSet, DefIndex};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_infer::infer::error_reporting::TypeAnnotationNeeded::E0282;
use rustc_infer::infer::InferCtxt;
use rustc_span::symbol::sym;
use rustc_span::Span;
use rustc_trait_selection::opaque_types::InferCtxtExt;

use std::mem;

///////////////////////////////////////////////////////////////////////////
// Entry point

// During type inference, partially inferred types are
// represented using Type variables (ty::Infer). These don't appear in
// the final TypeckTables since all of the types should have been
// inferred once typeck_tables_of is done.
// When type inference is running however, having to update the typeck
// tables every time a new type is inferred would be unreasonably slow,
// so instead all of the replacement happens at the end in
// resolve_type_vars_in_body, which creates a new TypeTables which
// doesn't contain any inference types.
impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn resolve_type_vars_in_body(
        &self,
        body: &'tcx hir::Body<'tcx>,
    ) -> &'tcx ty::TypeckTables<'tcx> {
        let item_id = self.tcx.hir().body_owner(body.id());
        let item_def_id = self.tcx.hir().local_def_id(item_id);

        // This attribute causes us to dump some writeback information
        // in the form of errors, which is uSymbolfor unit tests.
        let rustc_dump_user_substs = self.tcx.has_attr(item_def_id, sym::rustc_dump_user_substs);

        let mut wbcx = WritebackCx::new(self, body, rustc_dump_user_substs);
        for param in body.params {
            wbcx.visit_node_id(param.pat.span, param.hir_id);
        }
        // Type only exists for constants and statics, not functions.
        match self.tcx.hir().body_owner_kind(item_id) {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => {
                wbcx.visit_node_id(body.value.span, item_id);
            }
            hir::BodyOwnerKind::Closure | hir::BodyOwnerKind::Fn => (),
        }
        wbcx.visit_body(body);
        wbcx.visit_upvar_capture_map();
        wbcx.visit_closures();
        wbcx.visit_liberated_fn_sigs();
        wbcx.visit_fru_field_types();
        wbcx.visit_opaque_types(body.value.span);
        wbcx.visit_coercion_casts();
        wbcx.visit_free_region_map();
        wbcx.visit_user_provided_tys();
        wbcx.visit_user_provided_sigs();
        wbcx.visit_generator_interior_types();

        let used_trait_imports = mem::replace(
            &mut self.tables.borrow_mut().used_trait_imports,
            Lrc::new(DefIdSet::default()),
        );
        debug!("used_trait_imports({:?}) = {:?}", item_def_id, used_trait_imports);
        wbcx.tables.used_trait_imports = used_trait_imports;

        wbcx.tables.upvar_list =
            mem::replace(&mut self.tables.borrow_mut().upvar_list, Default::default());

        wbcx.tables.tainted_by_errors = self.is_tainted_by_errors();

        debug!("writeback: tables for {:?} are {:#?}", item_def_id, wbcx.tables);

        self.tcx.arena.alloc(wbcx.tables)
    }
}

///////////////////////////////////////////////////////////////////////////
// The Writeback context. This visitor walks the AST, checking the
// fn-specific tables to find references to types or regions. It
// resolves those regions to remove inference variables and writes the
// final result back into the master tables in the tcx. Here and
// there, it applies a few ad-hoc checks that were not convenient to
// do elsewhere.

struct WritebackCx<'cx, 'tcx> {
    fcx: &'cx FnCtxt<'cx, 'tcx>,

    tables: ty::TypeckTables<'tcx>,

    body: &'tcx hir::Body<'tcx>,

    rustc_dump_user_substs: bool,
}

impl<'cx, 'tcx> WritebackCx<'cx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'tcx>,
        body: &'tcx hir::Body<'tcx>,
        rustc_dump_user_substs: bool,
    ) -> WritebackCx<'cx, 'tcx> {
        let owner = body.id().hir_id;

        WritebackCx {
            fcx,
            tables: ty::TypeckTables::empty(Some(DefId::local(owner.owner))),
            body,
            rustc_dump_user_substs,
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fcx.tcx
    }

    fn write_ty_to_tables(&mut self, hir_id: hir::HirId, ty: Ty<'tcx>) {
        debug!("write_ty_to_tables({:?}, {:?})", hir_id, ty);
        assert!(!ty.needs_infer() && !ty.has_placeholders());
        self.tables.node_types_mut().insert(hir_id, ty);
    }

    // Hacky hack: During type-checking, we treat *all* operators
    // as potentially overloaded. But then, during writeback, if
    // we observe that something like `a+b` is (known to be)
    // operating on scalars, we clear the overload.
    fn fix_scalar_builtin_expr(&mut self, e: &hir::Expr<'_>) {
        match e.kind {
            hir::ExprKind::Unary(hir::UnOp::UnNeg, ref inner)
            | hir::ExprKind::Unary(hir::UnOp::UnNot, ref inner) => {
                let inner_ty = self.fcx.node_ty(inner.hir_id);
                let inner_ty = self.fcx.resolve_vars_if_possible(&inner_ty);

                if inner_ty.is_scalar() {
                    let mut tables = self.fcx.tables.borrow_mut();
                    tables.type_dependent_defs_mut().remove(e.hir_id);
                    tables.node_substs_mut().remove(e.hir_id);
                }
            }
            hir::ExprKind::Binary(ref op, ref lhs, ref rhs)
            | hir::ExprKind::AssignOp(ref op, ref lhs, ref rhs) => {
                let lhs_ty = self.fcx.node_ty(lhs.hir_id);
                let lhs_ty = self.fcx.resolve_vars_if_possible(&lhs_ty);

                let rhs_ty = self.fcx.node_ty(rhs.hir_id);
                let rhs_ty = self.fcx.resolve_vars_if_possible(&rhs_ty);

                if lhs_ty.is_scalar() && rhs_ty.is_scalar() {
                    let mut tables = self.fcx.tables.borrow_mut();
                    tables.type_dependent_defs_mut().remove(e.hir_id);
                    tables.node_substs_mut().remove(e.hir_id);

                    match e.kind {
                        hir::ExprKind::Binary(..) => {
                            if !op.node.is_by_value() {
                                let mut adjustments = tables.adjustments_mut();
                                adjustments.get_mut(lhs.hir_id).map(|a| a.pop());
                                adjustments.get_mut(rhs.hir_id).map(|a| a.pop());
                            }
                        }
                        hir::ExprKind::AssignOp(..) => {
                            tables.adjustments_mut().get_mut(lhs.hir_id).map(|a| a.pop());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Similar to operators, indexing is always assumed to be overloaded
    // Here, correct cases where an indexing expression can be simplified
    // to use builtin indexing because the index type is known to be
    // usize-ish
    fn fix_index_builtin_expr(&mut self, e: &hir::Expr<'_>) {
        if let hir::ExprKind::Index(ref base, ref index) = e.kind {
            let mut tables = self.fcx.tables.borrow_mut();

            // All valid indexing looks like this; might encounter non-valid indexes at this point.
            let base_ty = tables.expr_ty_adjusted_opt(&base).map(|t| &t.kind);
            if base_ty.is_none() {
                // When encountering `return [0][0]` outside of a `fn` body we can encounter a base
                // that isn't in the type table. We assume more relevant errors have already been
                // emitted, so we delay an ICE if none have. (#64638)
                self.tcx().sess.delay_span_bug(e.span, &format!("bad base: `{:?}`", base));
            }
            if let Some(ty::Ref(_, base_ty, _)) = base_ty {
                let index_ty = tables.expr_ty_adjusted_opt(&index).unwrap_or_else(|| {
                    // When encountering `return [0][0]` outside of a `fn` body we would attempt
                    // to access an unexistend index. We assume that more relevant errors will
                    // already have been emitted, so we only gate on this with an ICE if no
                    // error has been emitted. (#64638)
                    self.tcx().sess.delay_span_bug(
                        e.span,
                        &format!("bad index {:?} for base: `{:?}`", index, base),
                    );
                    self.fcx.tcx.types.err
                });
                let index_ty = self.fcx.resolve_vars_if_possible(&index_ty);

                if base_ty.builtin_index().is_some() && index_ty == self.fcx.tcx.types.usize {
                    // Remove the method call record
                    tables.type_dependent_defs_mut().remove(e.hir_id);
                    tables.node_substs_mut().remove(e.hir_id);

                    tables.adjustments_mut().get_mut(base.hir_id).map(|a| {
                        // Discard the need for a mutable borrow
                        match a.pop() {
                            // Extra adjustment made when indexing causes a drop
                            // of size information - we need to get rid of it
                            // Since this is "after" the other adjustment to be
                            // discarded, we do an extra `pop()`
                            Some(Adjustment {
                                kind: Adjust::Pointer(PointerCast::Unsize), ..
                            }) => {
                                // So the borrow discard actually happens here
                                a.pop();
                            }
                            _ => {}
                        }
                    });
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Impl of Visitor for Resolver
//
// This is the master code which walks the AST. It delegates most of
// the heavy lifting to the generic visit and resolve functions
// below. In general, a function is made into a `visitor` if it must
// traffic in node-ids or update tables in the type context etc.

impl<'cx, 'tcx> Visitor<'tcx> for WritebackCx<'cx, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        self.fix_scalar_builtin_expr(e);
        self.fix_index_builtin_expr(e);

        self.visit_node_id(e.span, e.hir_id);

        match e.kind {
            hir::ExprKind::Closure(_, _, body, _, _) => {
                let body = self.fcx.tcx.hir().body(body);
                for param in body.params {
                    self.visit_node_id(e.span, param.hir_id);
                }

                self.visit_body(body);
            }
            hir::ExprKind::Struct(_, fields, _) => {
                for field in fields {
                    self.visit_field_id(field.hir_id);
                }
            }
            hir::ExprKind::Field(..) => {
                self.visit_field_id(e.hir_id);
            }
            _ => {}
        }

        intravisit::walk_expr(self, e);
    }

    fn visit_block(&mut self, b: &'tcx hir::Block<'tcx>) {
        self.visit_node_id(b.span, b.hir_id);
        intravisit::walk_block(self, b);
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        match p.kind {
            hir::PatKind::Binding(..) => {
                let tables = self.fcx.tables.borrow();
                if let Some(bm) = tables.extract_binding_mode(self.tcx().sess, p.hir_id, p.span) {
                    self.tables.pat_binding_modes_mut().insert(p.hir_id, bm);
                }
            }
            hir::PatKind::Struct(_, fields, _) => {
                for field in fields {
                    self.visit_field_id(field.hir_id);
                }
            }
            _ => {}
        };

        self.visit_pat_adjustments(p.span, p.hir_id);

        self.visit_node_id(p.span, p.hir_id);
        intravisit::walk_pat(self, p);
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        intravisit::walk_local(self, l);
        let var_ty = self.fcx.local_ty(l.span, l.hir_id).decl_ty;
        let var_ty = self.resolve(&var_ty, &l.span);
        self.write_ty_to_tables(l.hir_id, var_ty);
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty<'tcx>) {
        intravisit::walk_ty(self, hir_ty);
        let ty = self.fcx.node_ty(hir_ty.hir_id);
        let ty = self.resolve(&ty, &hir_ty.span);
        self.write_ty_to_tables(hir_ty.hir_id, ty);
    }
}

impl<'cx, 'tcx> WritebackCx<'cx, 'tcx> {
    fn visit_upvar_capture_map(&mut self) {
        for (upvar_id, upvar_capture) in self.fcx.tables.borrow().upvar_capture_map.iter() {
            let new_upvar_capture = match *upvar_capture {
                ty::UpvarCapture::ByValue => ty::UpvarCapture::ByValue,
                ty::UpvarCapture::ByRef(ref upvar_borrow) => {
                    let r = upvar_borrow.region;
                    let r = self.resolve(&r, &upvar_id.var_path.hir_id);
                    ty::UpvarCapture::ByRef(ty::UpvarBorrow { kind: upvar_borrow.kind, region: r })
                }
            };
            debug!("Upvar capture for {:?} resolved to {:?}", upvar_id, new_upvar_capture);
            self.tables.upvar_capture_map.insert(*upvar_id, new_upvar_capture);
        }
    }

    fn visit_closures(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&id, &origin) in fcx_tables.closure_kind_origins().iter() {
            let hir_id = hir::HirId { owner: common_local_id_root.index, local_id: id };
            self.tables.closure_kind_origins_mut().insert(hir_id, origin);
        }
    }

    fn visit_coercion_casts(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        let fcx_coercion_casts = fcx_tables.coercion_casts();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);

        for local_id in fcx_coercion_casts {
            self.tables.set_coercion_cast(*local_id);
        }
    }

    fn visit_free_region_map(&mut self) {
        self.tables.free_region_map = self.fcx.tables.borrow().free_region_map.clone();
        debug_assert!(!self.tables.free_region_map.elements().any(|r| r.has_local_value()));
    }

    fn visit_user_provided_tys(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        let mut errors_buffer = Vec::new();
        for (&local_id, c_ty) in fcx_tables.user_provided_types().iter() {
            let hir_id = hir::HirId { owner: common_local_id_root.index, local_id };

            if cfg!(debug_assertions) && c_ty.has_local_value() {
                span_bug!(hir_id.to_span(self.fcx.tcx), "writeback: `{:?}` is a local value", c_ty);
            };

            self.tables.user_provided_types_mut().insert(hir_id, c_ty.clone());

            if let ty::UserType::TypeOf(_, user_substs) = c_ty.value {
                if self.rustc_dump_user_substs {
                    // This is a unit-testing mechanism.
                    let span = self.tcx().hir().span(hir_id);
                    // We need to buffer the errors in order to guarantee a consistent
                    // order when emitting them.
                    let err = self
                        .tcx()
                        .sess
                        .struct_span_err(span, &format!("user substs: {:?}", user_substs));
                    err.buffer(&mut errors_buffer);
                }
            }
        }

        if !errors_buffer.is_empty() {
            errors_buffer.sort_by_key(|diag| diag.span.primary_span());
            for diag in errors_buffer.drain(..) {
                self.tcx().sess.diagnostic().emit_diagnostic(&diag);
            }
        }
    }

    fn visit_user_provided_sigs(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);

        for (&def_id, c_sig) in fcx_tables.user_provided_sigs.iter() {
            if cfg!(debug_assertions) && c_sig.has_local_value() {
                span_bug!(
                    self.fcx.tcx.hir().span_if_local(def_id).unwrap(),
                    "writeback: `{:?}` is a local value",
                    c_sig
                );
            };

            self.tables.user_provided_sigs.insert(def_id, c_sig.clone());
        }
    }

    fn visit_generator_interior_types(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        self.tables.generator_interior_types = fcx_tables.generator_interior_types.clone();
    }

    fn visit_opaque_types(&mut self, span: Span) {
        for (&def_id, opaque_defn) in self.fcx.opaque_types.borrow().iter() {
            let hir_id = self.tcx().hir().as_local_hir_id(def_id).unwrap();
            let instantiated_ty =
                self.tcx().erase_regions(&self.resolve(&opaque_defn.concrete_ty, &hir_id));

            debug_assert!(!instantiated_ty.has_escaping_bound_vars());

            // Prevent:
            // * `fn foo<T>() -> Foo<T>`
            // * `fn foo<T: Bound + Other>() -> Foo<T>`
            // from being defining.

            // Also replace all generic params with the ones from the opaque type
            // definition so that
            // ```rust
            // type Foo<T> = impl Baz + 'static;
            // fn foo<U>() -> Foo<U> { .. }
            // ```
            // figures out the concrete type with `U`, but the stored type is with `T`.
            let definition_ty = self.fcx.infer_opaque_definition_from_instantiation(
                def_id,
                opaque_defn.substs,
                instantiated_ty,
                span,
            );

            let mut skip_add = false;

            if let ty::Opaque(defin_ty_def_id, _substs) = definition_ty.kind {
                if let hir::OpaqueTyOrigin::TypeAlias = opaque_defn.origin {
                    if def_id == defin_ty_def_id {
                        debug!(
                            "skipping adding concrete definition for opaque type {:?} {:?}",
                            opaque_defn, defin_ty_def_id
                        );
                        skip_add = true;
                    }
                }
            }

            if !opaque_defn.substs.has_local_value() {
                // We only want to add an entry into `concrete_opaque_types`
                // if we actually found a defining usage of this opaque type.
                // Otherwise, we do nothing - we'll either find a defining usage
                // in some other location, or we'll end up emitting an error due
                // to the lack of defining usage
                if !skip_add {
                    let new = ty::ResolvedOpaqueTy {
                        concrete_type: definition_ty,
                        substs: opaque_defn.substs,
                    };

                    let old = self.tables.concrete_opaque_types.insert(def_id, new);
                    if let Some(old) = old {
                        if old.concrete_type != definition_ty || old.substs != opaque_defn.substs {
                            span_bug!(
                                span,
                                "`visit_opaque_types` tried to write different types for the same \
                                 opaque type: {:?}, {:?}, {:?}, {:?}",
                                def_id,
                                definition_ty,
                                opaque_defn,
                                old,
                            );
                        }
                    }
                }
            } else {
                self.tcx().sess.delay_span_bug(span, "`opaque_defn` is a local value");
            }
        }
    }

    fn visit_field_id(&mut self, hir_id: hir::HirId) {
        if let Some(index) = self.fcx.tables.borrow_mut().field_indices_mut().remove(hir_id) {
            self.tables.field_indices_mut().insert(hir_id, index);
        }
    }

    fn visit_node_id(&mut self, span: Span, hir_id: hir::HirId) {
        // Export associated path extensions and method resolutions.
        if let Some(def) = self.fcx.tables.borrow_mut().type_dependent_defs_mut().remove(hir_id) {
            self.tables.type_dependent_defs_mut().insert(hir_id, def);
        }

        // Resolve any borrowings for the node with id `node_id`
        self.visit_adjustments(span, hir_id);

        // Resolve the type of the node with id `node_id`
        let n_ty = self.fcx.node_ty(hir_id);
        let n_ty = self.resolve(&n_ty, &span);
        self.write_ty_to_tables(hir_id, n_ty);
        debug!("node {:?} has type {:?}", hir_id, n_ty);

        // Resolve any substitutions
        if let Some(substs) = self.fcx.tables.borrow().node_substs_opt(hir_id) {
            let substs = self.resolve(&substs, &span);
            debug!("write_substs_to_tcx({:?}, {:?})", hir_id, substs);
            assert!(!substs.needs_infer() && !substs.has_placeholders());
            self.tables.node_substs_mut().insert(hir_id, substs);
        }
    }

    fn visit_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
        let adjustment = self.fcx.tables.borrow_mut().adjustments_mut().remove(hir_id);
        match adjustment {
            None => {
                debug!("no adjustments for node {:?}", hir_id);
            }

            Some(adjustment) => {
                let resolved_adjustment = self.resolve(&adjustment, &span);
                debug!("adjustments for node {:?}: {:?}", hir_id, resolved_adjustment);
                self.tables.adjustments_mut().insert(hir_id, resolved_adjustment);
            }
        }
    }

    fn visit_pat_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
        let adjustment = self.fcx.tables.borrow_mut().pat_adjustments_mut().remove(hir_id);
        match adjustment {
            None => {
                debug!("no pat_adjustments for node {:?}", hir_id);
            }

            Some(adjustment) => {
                let resolved_adjustment = self.resolve(&adjustment, &span);
                debug!("pat_adjustments for node {:?}: {:?}", hir_id, resolved_adjustment);
                self.tables.pat_adjustments_mut().insert(hir_id, resolved_adjustment);
            }
        }
    }

    fn visit_liberated_fn_sigs(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&local_id, fn_sig) in fcx_tables.liberated_fn_sigs().iter() {
            let hir_id = hir::HirId { owner: common_local_id_root.index, local_id };
            let fn_sig = self.resolve(fn_sig, &hir_id);
            self.tables.liberated_fn_sigs_mut().insert(hir_id, fn_sig.clone());
        }
    }

    fn visit_fru_field_types(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&local_id, ftys) in fcx_tables.fru_field_types().iter() {
            let hir_id = hir::HirId { owner: common_local_id_root.index, local_id };
            let ftys = self.resolve(ftys, &hir_id);
            self.tables.fru_field_types_mut().insert(hir_id, ftys);
        }
    }

    fn resolve<T>(&self, x: &T, span: &dyn Locatable) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let x = x.fold_with(&mut Resolver::new(self.fcx, span, self.body));
        if cfg!(debug_assertions) && x.has_local_value() {
            span_bug!(span.to_span(self.fcx.tcx), "writeback: `{:?}` is a local value", x);
        }
        x
    }
}

trait Locatable {
    fn to_span(&self, tcx: TyCtxt<'_>) -> Span;
}

impl Locatable for Span {
    fn to_span(&self, _: TyCtxt<'_>) -> Span {
        *self
    }
}

impl Locatable for DefIndex {
    fn to_span(&self, tcx: TyCtxt<'_>) -> Span {
        let hir_id = tcx.hir().def_index_to_hir_id(*self);
        tcx.hir().span(hir_id)
    }
}

impl Locatable for hir::HirId {
    fn to_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(*self)
    }
}

///////////////////////////////////////////////////////////////////////////
// The Resolver. This is the type folding engine that detects
// unresolved types and so forth.

struct Resolver<'cx, 'tcx> {
    tcx: TyCtxt<'tcx>,
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    span: &'cx dyn Locatable,
    body: &'tcx hir::Body<'tcx>,
}

impl<'cx, 'tcx> Resolver<'cx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'tcx>,
        span: &'cx dyn Locatable,
        body: &'tcx hir::Body<'tcx>,
    ) -> Resolver<'cx, 'tcx> {
        Resolver { tcx: fcx.tcx, infcx: fcx, span, body }
    }

    fn report_error(&self, t: Ty<'tcx>) {
        if !self.tcx.sess.has_errors() {
            self.infcx
                .need_type_info_err(Some(self.body.id()), self.span.to_span(self.tcx), t, E0282)
                .emit();
        }
    }
}

impl<'cx, 'tcx> TypeFolder<'tcx> for Resolver<'cx, 'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match self.infcx.fully_resolve(&t) {
            Ok(t) => t,
            Err(_) => {
                debug!("Resolver::fold_ty: input type `{:?}` not fully resolvable", t);
                self.report_error(t);
                self.tcx().types.err
            }
        }
    }

    // FIXME This should be carefully checked
    // We could use `self.report_error` but it doesn't accept a ty::Region, right now.
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        self.infcx.fully_resolve(&r).unwrap_or(self.tcx.lifetimes.re_static)
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        match self.infcx.fully_resolve(&ct) {
            Ok(ct) => ct,
            Err(_) => {
                debug!("Resolver::fold_const: input const `{:?}` not fully resolvable", ct);
                // FIXME: we'd like to use `self.report_error`, but it doesn't yet
                // accept a &'tcx ty::Const.
                self.tcx().consts.err
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// During type check, we store promises with the result of trait
// lookup rather than the actual results (because the results are not
// necessarily available immediately). These routines unwind the
// promises. It is expected that we will have already reported any
// errors that may be encountered, so if the promises store an error,
// a dummy result is returned.
