// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.

use crate::check::FnCtxt;
use errors::DiagnosticBuilder;
use rustc::hir;
use rustc::hir::def_id::{DefId, DefIndex};
use rustc::hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc::infer::InferCtxt;
use rustc::ty::adjustment::{Adjust, Adjustment};
use rustc::ty::fold::{BottomUpFolder, TypeFoldable, TypeFolder};
use rustc::ty::subst::UnpackedKind;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::util::nodemap::DefIdSet;
use rustc_data_structures::sync::Lrc;
use std::mem;
use syntax::ast;
use syntax_pos::Span;

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
impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn resolve_type_vars_in_body(&self, body: &'gcx hir::Body) -> &'gcx ty::TypeckTables<'gcx> {
        let item_id = self.tcx.hir().body_owner(body.id());
        let item_def_id = self.tcx.hir().local_def_id(item_id);

        // This attribute causes us to dump some writeback information
        // in the form of errors, which is used for unit tests.
        let rustc_dump_user_substs = self.tcx.has_attr(item_def_id, "rustc_dump_user_substs");

        let mut wbcx = WritebackCx::new(self, body, rustc_dump_user_substs);
        for arg in &body.arguments {
            wbcx.visit_node_id(arg.pat.span, arg.hir_id);
        }
        wbcx.visit_body(body);
        wbcx.visit_upvar_capture_map();
        wbcx.visit_upvar_list_map();
        wbcx.visit_closures();
        wbcx.visit_liberated_fn_sigs();
        wbcx.visit_fru_field_types();
        wbcx.visit_opaque_types(body.value.span);
        wbcx.visit_cast_types();
        wbcx.visit_free_region_map();
        wbcx.visit_user_provided_tys();
        wbcx.visit_user_provided_sigs();

        let used_trait_imports = mem::replace(
            &mut self.tables.borrow_mut().used_trait_imports,
            Lrc::new(DefIdSet::default()),
        );
        debug!(
            "used_trait_imports({:?}) = {:?}",
            item_def_id, used_trait_imports
        );
        wbcx.tables.used_trait_imports = used_trait_imports;

        wbcx.tables.tainted_by_errors = self.is_tainted_by_errors();

        debug!(
            "writeback: tables for {:?} are {:#?}",
            item_def_id, wbcx.tables
        );

        self.tcx.alloc_tables(wbcx.tables)
    }
}

///////////////////////////////////////////////////////////////////////////
// The Writeback context. This visitor walks the AST, checking the
// fn-specific tables to find references to types or regions. It
// resolves those regions to remove inference variables and writes the
// final result back into the master tables in the tcx. Here and
// there, it applies a few ad-hoc checks that were not convenient to
// do elsewhere.

struct WritebackCx<'cx, 'gcx: 'cx + 'tcx, 'tcx: 'cx> {
    fcx: &'cx FnCtxt<'cx, 'gcx, 'tcx>,

    tables: ty::TypeckTables<'gcx>,

    body: &'gcx hir::Body,

    rustc_dump_user_substs: bool,
}

impl<'cx, 'gcx, 'tcx> WritebackCx<'cx, 'gcx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'gcx, 'tcx>,
        body: &'gcx hir::Body,
        rustc_dump_user_substs: bool,
    ) -> WritebackCx<'cx, 'gcx, 'tcx> {
        let owner = body.id().hir_id;

        WritebackCx {
            fcx,
            tables: ty::TypeckTables::empty(Some(DefId::local(owner.owner))),
            body,
            rustc_dump_user_substs,
        }
    }

    fn tcx(&self) -> TyCtxt<'cx, 'gcx, 'tcx> {
        self.fcx.tcx
    }

    fn write_ty_to_tables(&mut self, hir_id: hir::HirId, ty: Ty<'gcx>) {
        debug!("write_ty_to_tables({:?}, {:?})", hir_id, ty);
        assert!(!ty.needs_infer() && !ty.has_placeholders());
        self.tables.node_types_mut().insert(hir_id, ty);
    }

    // Hacky hack: During type-checking, we treat *all* operators
    // as potentially overloaded. But then, during writeback, if
    // we observe that something like `a+b` is (known to be)
    // operating on scalars, we clear the overload.
    fn fix_scalar_builtin_expr(&mut self, e: &hir::Expr) {
        match e.node {
            hir::ExprKind::Unary(hir::UnNeg, ref inner)
            | hir::ExprKind::Unary(hir::UnNot, ref inner) => {
                let inner_ty = self.fcx.node_ty(inner.hir_id);
                let inner_ty = self.fcx.resolve_type_vars_if_possible(&inner_ty);

                if inner_ty.is_scalar() {
                    let mut tables = self.fcx.tables.borrow_mut();
                    tables.type_dependent_defs_mut().remove(e.hir_id);
                    tables.node_substs_mut().remove(e.hir_id);
                }
            }
            hir::ExprKind::Binary(ref op, ref lhs, ref rhs)
            | hir::ExprKind::AssignOp(ref op, ref lhs, ref rhs) => {
                let lhs_ty = self.fcx.node_ty(lhs.hir_id);
                let lhs_ty = self.fcx.resolve_type_vars_if_possible(&lhs_ty);

                let rhs_ty = self.fcx.node_ty(rhs.hir_id);
                let rhs_ty = self.fcx.resolve_type_vars_if_possible(&rhs_ty);

                if lhs_ty.is_scalar() && rhs_ty.is_scalar() {
                    let mut tables = self.fcx.tables.borrow_mut();
                    tables.type_dependent_defs_mut().remove(e.hir_id);
                    tables.node_substs_mut().remove(e.hir_id);

                    match e.node {
                        hir::ExprKind::Binary(..) => {
                            if !op.node.is_by_value() {
                                let mut adjustments = tables.adjustments_mut();
                                adjustments.get_mut(lhs.hir_id).map(|a| a.pop());
                                adjustments.get_mut(rhs.hir_id).map(|a| a.pop());
                            }
                        }
                        hir::ExprKind::AssignOp(..) => {
                            tables
                                .adjustments_mut()
                                .get_mut(lhs.hir_id)
                                .map(|a| a.pop());
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
    fn fix_index_builtin_expr(&mut self, e: &hir::Expr) {
        if let hir::ExprKind::Index(ref base, ref index) = e.node {
            let mut tables = self.fcx.tables.borrow_mut();

            // All valid indexing looks like this; might encounter non-valid indexes at this point
            if let ty::Ref(_, base_ty, _) = tables.expr_ty_adjusted(&base).sty {
                let index_ty = tables.expr_ty_adjusted(&index);
                let index_ty = self.fcx.resolve_type_vars_if_possible(&index_ty);

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
                                kind: Adjust::Unsize,
                                ..
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

impl<'cx, 'gcx, 'tcx> Visitor<'gcx> for WritebackCx<'cx, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, e: &'gcx hir::Expr) {
        self.fix_scalar_builtin_expr(e);
        self.fix_index_builtin_expr(e);

        self.visit_node_id(e.span, e.hir_id);

        match e.node {
            hir::ExprKind::Closure(_, _, body, _, _) => {
                let body = self.fcx.tcx.hir().body(body);
                for arg in &body.arguments {
                    self.visit_node_id(e.span, arg.hir_id);
                }

                self.visit_body(body);
            }
            hir::ExprKind::Struct(_, ref fields, _) => {
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

    fn visit_block(&mut self, b: &'gcx hir::Block) {
        self.visit_node_id(b.span, b.hir_id);
        intravisit::walk_block(self, b);
    }

    fn visit_pat(&mut self, p: &'gcx hir::Pat) {
        match p.node {
            hir::PatKind::Binding(..) => {
                if let Some(&bm) = self.fcx.tables.borrow().pat_binding_modes().get(p.hir_id) {
                    self.tables.pat_binding_modes_mut().insert(p.hir_id, bm);
                } else {
                    self.tcx()
                        .sess
                        .delay_span_bug(p.span, "missing binding mode");
                }
            }
            hir::PatKind::Struct(_, ref fields, _) => {
                for field in fields {
                    self.visit_field_id(field.node.hir_id);
                }
            }
            _ => {}
        };

        self.visit_pat_adjustments(p.span, p.hir_id);

        self.visit_node_id(p.span, p.hir_id);
        intravisit::walk_pat(self, p);
    }

    fn visit_local(&mut self, l: &'gcx hir::Local) {
        intravisit::walk_local(self, l);
        let var_ty = self.fcx.local_ty(l.span, l.hir_id).decl_ty;
        let var_ty = self.resolve(&var_ty, &l.span);
        self.write_ty_to_tables(l.hir_id, var_ty);
    }

    fn visit_ty(&mut self, hir_ty: &'gcx hir::Ty) {
        intravisit::walk_ty(self, hir_ty);
        let ty = self.fcx.node_ty(hir_ty.hir_id);
        let ty = self.resolve(&ty, &hir_ty.span);
        self.write_ty_to_tables(hir_ty.hir_id, ty);
    }
}

impl<'cx, 'gcx, 'tcx> WritebackCx<'cx, 'gcx, 'tcx> {
    fn visit_upvar_capture_map(&mut self) {
        for (upvar_id, upvar_capture) in self.fcx.tables.borrow().upvar_capture_map.iter() {
            let new_upvar_capture = match *upvar_capture {
                ty::UpvarCapture::ByValue => ty::UpvarCapture::ByValue,
                ty::UpvarCapture::ByRef(ref upvar_borrow) => {
                    let r = upvar_borrow.region;
                    let r = self.resolve(&r, &upvar_id.var_path.hir_id);
                    ty::UpvarCapture::ByRef(ty::UpvarBorrow {
                        kind: upvar_borrow.kind,
                        region: r,
                    })
                }
            };
            debug!(
                "Upvar capture for {:?} resolved to {:?}",
                upvar_id, new_upvar_capture
            );
            self.tables
                .upvar_capture_map
                .insert(*upvar_id, new_upvar_capture);
        }
    }

    /// Runs through the function context's upvar list map and adds the same to
    /// the TypeckTables. upvarlist is a hashmap of the list of upvars referred
    /// to in a closure..
    fn visit_upvar_list_map(&mut self) {
        for (closure_def_id, upvar_list) in self.fcx.tables.borrow().upvar_list.iter() {
            debug!(
                "UpvarIDs captured by closure {:?} are: {:?}",
                closure_def_id, upvar_list
            );
            self.tables
                .upvar_list
                .insert(*closure_def_id, upvar_list.to_vec());
        }
    }

    fn visit_closures(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&id, &origin) in fcx_tables.closure_kind_origins().iter() {
            let hir_id = hir::HirId {
                owner: common_local_id_root.index,
                local_id: id,
            };
            self.tables
                .closure_kind_origins_mut()
                .insert(hir_id, origin);
        }
    }

    fn visit_cast_types(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        let fcx_cast_kinds = fcx_tables.cast_kinds();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let mut self_cast_kinds = self.tables.cast_kinds_mut();
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&local_id, &cast_kind) in fcx_cast_kinds.iter() {
            let hir_id = hir::HirId {
                owner: common_local_id_root.index,
                local_id,
            };
            self_cast_kinds.insert(hir_id, cast_kind);
        }
    }

    fn visit_free_region_map(&mut self) {
        let free_region_map = self.tcx()
            .lift_to_global(&self.fcx.tables.borrow().free_region_map);
        let free_region_map = free_region_map.expect("all regions in free-region-map are global");
        self.tables.free_region_map = free_region_map;
    }

    fn visit_user_provided_tys(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        let mut errors_buffer = Vec::new();
        for (&local_id, c_ty) in fcx_tables.user_provided_types().iter() {
            let hir_id = hir::HirId {
                owner: common_local_id_root.index,
                local_id,
            };

            let c_ty = if let Some(c_ty) = self.tcx().lift_to_global(c_ty) {
                c_ty
            } else {
                span_bug!(
                    hir_id.to_span(&self.fcx.tcx),
                    "writeback: `{:?}` missing from the global type context",
                    c_ty
                );
            };

            self.tables
                .user_provided_types_mut()
                .insert(hir_id, c_ty.clone());

            if let ty::UserType::TypeOf(_, user_substs) = c_ty.value {
                if self.rustc_dump_user_substs {
                    // This is a unit-testing mechanism.
                    let span = self.tcx().hir().span_by_hir_id(hir_id);
                    // We need to buffer the errors in order to guarantee a consistent
                    // order when emitting them.
                    let err = self.tcx().sess.struct_span_err(
                        span,
                        &format!("user substs: {:?}", user_substs)
                    );
                    err.buffer(&mut errors_buffer);
                }
            }
        }

        if !errors_buffer.is_empty() {
            errors_buffer.sort_by_key(|diag| diag.span.primary_span());
            for diag in errors_buffer.drain(..) {
                DiagnosticBuilder::new_diagnostic(self.tcx().sess.diagnostic(), diag).emit();
            }
        }
    }

    fn visit_user_provided_sigs(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);

        for (&def_id, c_sig) in fcx_tables.user_provided_sigs.iter() {
            let c_sig = if let Some(c_sig) = self.tcx().lift_to_global(c_sig) {
                c_sig
            } else {
                span_bug!(
                    self.fcx.tcx.hir().span_if_local(def_id).unwrap(),
                    "writeback: `{:?}` missing from the global type context",
                    c_sig
                );
            };

            self.tables
                .user_provided_sigs
                .insert(def_id, c_sig.clone());
        }
    }

    fn visit_opaque_types(&mut self, span: Span) {
        for (&def_id, opaque_defn) in self.fcx.opaque_types.borrow().iter() {
            let node_id = self.tcx().hir().as_local_node_id(def_id).unwrap();
            let instantiated_ty = self.resolve(&opaque_defn.concrete_ty, &node_id);

            let generics = self.tcx().generics_of(def_id);

            let definition_ty = if generics.parent.is_some() {
                // impl trait
                self.fcx.infer_opaque_definition_from_instantiation(
                    def_id,
                    opaque_defn,
                    instantiated_ty,
                )
            } else {
                // prevent
                // * `fn foo<T>() -> Foo<T>`
                // * `fn foo<T: Bound + Other>() -> Foo<T>`
                // from being defining

                // Also replace all generic params with the ones from the existential type
                // definition so
                // ```rust
                // existential type Foo<T>: 'static;
                // fn foo<U>() -> Foo<U> { .. }
                // ```
                // figures out the concrete type with `U`, but the stored type is with `T`
                instantiated_ty.fold_with(&mut BottomUpFolder {
                    tcx: self.tcx().global_tcx(),
                    fldop: |ty| {
                        trace!("checking type {:?}: {:#?}", ty, ty.sty);
                        // find a type parameter
                        if let ty::Param(..) = ty.sty {
                            // look it up in the substitution list
                            assert_eq!(opaque_defn.substs.len(), generics.params.len());
                            for (subst, param) in opaque_defn.substs.iter().zip(&generics.params) {
                                if let UnpackedKind::Type(subst) = subst.unpack() {
                                    if subst == ty {
                                        // found it in the substitution list, replace with the
                                        // parameter from the existential type
                                        return self.tcx()
                                            .global_tcx()
                                            .mk_ty_param(param.index, param.name);
                                    }
                                }
                            }
                            self.tcx()
                                .sess
                                .struct_span_err(
                                    span,
                                    &format!(
                                        "type parameter `{}` is part of concrete type but not used \
                                         in parameter list for existential type",
                                        ty,
                                    ),
                                )
                                .emit();
                            return self.tcx().types.err;
                        }
                        ty
                    },
                    reg_op: |region| {
                        match region {
                            // ignore static regions
                            ty::ReStatic => region,
                            _ => {
                                trace!("checking {:?}", region);
                                for (subst, p) in opaque_defn.substs.iter().zip(&generics.params) {
                                    if let UnpackedKind::Lifetime(subst) = subst.unpack() {
                                        if subst == region {
                                            // found it in the substitution list, replace with the
                                            // parameter from the existential type
                                            let reg = ty::EarlyBoundRegion {
                                                def_id: p.def_id,
                                                index: p.index,
                                                name: p.name,
                                            };
                                            trace!("replace {:?} with {:?}", region, reg);
                                            return self.tcx()
                                                .global_tcx()
                                                .mk_region(ty::ReEarlyBound(reg));
                                        }
                                    }
                                }
                                trace!("opaque_defn: {:#?}", opaque_defn);
                                trace!("generics: {:#?}", generics);
                                self.tcx()
                                    .sess
                                    .struct_span_err(
                                        span,
                                        "non-defining existential type use in defining scope",
                                    )
                                    .span_label(
                                        span,
                                        format!(
                                            "lifetime `{}` is part of concrete type but not used \
                                             in parameter list of existential type",
                                            region,
                                        ),
                                    )
                                    .emit();
                                self.tcx().global_tcx().mk_region(ty::ReStatic)
                            }
                        }
                    },
                }).unwrap_or_else(|e: !| e)
            };

            if let ty::Opaque(defin_ty_def_id, _substs) = definition_ty.sty {
                if def_id == defin_ty_def_id {
                    // Concrete type resolved to the existential type itself
                    // Force a cycle error
                    // FIXME(oli-obk): we could just not insert it into `concrete_existential_types`
                    // which simply would make this use not a defining use.
                    self.tcx().at(span).type_of(defin_ty_def_id);
                }
            }

            let new = ty::ResolvedOpaqueTy {
                concrete_type: definition_ty,
                substs: self.tcx().lift_to_global(&opaque_defn.substs).unwrap(),
            };

            let old = self.tables
                .concrete_existential_types
                .insert(def_id, new);
            if let Some(old) = old {
                if old.concrete_type != definition_ty || old.substs != opaque_defn.substs {
                    span_bug!(
                        span,
                        "visit_opaque_types tried to write \
                        different types for the same existential type: {:?}, {:?}, {:?}, {:?}",
                        def_id,
                        definition_ty,
                        opaque_defn,
                        old,
                    );
                }
            }
        }
    }

    fn visit_field_id(&mut self, hir_id: hir::HirId) {
        if let Some(index) = self.fcx
            .tables
            .borrow_mut()
            .field_indices_mut()
            .remove(hir_id)
        {
            self.tables.field_indices_mut().insert(hir_id, index);
        }
    }

    fn visit_node_id(&mut self, span: Span, hir_id: hir::HirId) {
        // Export associated path extensions and method resolutions.
        if let Some(def) = self.fcx
            .tables
            .borrow_mut()
            .type_dependent_defs_mut()
            .remove(hir_id)
        {
            self.tables.type_dependent_defs_mut().insert(hir_id, def);
        }

        // Resolve any borrowings for the node with id `node_id`
        self.visit_adjustments(span, hir_id);

        // Resolve the type of the node with id `node_id`
        let n_ty = self.fcx.node_ty(hir_id);
        let n_ty = self.resolve(&n_ty, &span);
        self.write_ty_to_tables(hir_id, n_ty);
        debug!("Node {:?} has type {:?}", hir_id, n_ty);

        // Resolve any substitutions
        if let Some(substs) = self.fcx.tables.borrow().node_substs_opt(hir_id) {
            let substs = self.resolve(&substs, &span);
            debug!("write_substs_to_tcx({:?}, {:?})", hir_id, substs);
            assert!(!substs.needs_infer() && !substs.has_placeholders());
            self.tables.node_substs_mut().insert(hir_id, substs);
        }
    }

    fn visit_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
        let adjustment = self.fcx
            .tables
            .borrow_mut()
            .adjustments_mut()
            .remove(hir_id);
        match adjustment {
            None => {
                debug!("No adjustments for node {:?}", hir_id);
            }

            Some(adjustment) => {
                let resolved_adjustment = self.resolve(&adjustment, &span);
                debug!(
                    "Adjustments for node {:?}: {:?}",
                    hir_id, resolved_adjustment
                );
                self.tables
                    .adjustments_mut()
                    .insert(hir_id, resolved_adjustment);
            }
        }
    }

    fn visit_pat_adjustments(&mut self, span: Span, hir_id: hir::HirId) {
        let adjustment = self.fcx
            .tables
            .borrow_mut()
            .pat_adjustments_mut()
            .remove(hir_id);
        match adjustment {
            None => {
                debug!("No pat_adjustments for node {:?}", hir_id);
            }

            Some(adjustment) => {
                let resolved_adjustment = self.resolve(&adjustment, &span);
                debug!(
                    "pat_adjustments for node {:?}: {:?}",
                    hir_id, resolved_adjustment
                );
                self.tables
                    .pat_adjustments_mut()
                    .insert(hir_id, resolved_adjustment);
            }
        }
    }

    fn visit_liberated_fn_sigs(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&local_id, fn_sig) in fcx_tables.liberated_fn_sigs().iter() {
            let hir_id = hir::HirId {
                owner: common_local_id_root.index,
                local_id,
            };
            let fn_sig = self.resolve(fn_sig, &hir_id);
            self.tables
                .liberated_fn_sigs_mut()
                .insert(hir_id, fn_sig.clone());
        }
    }

    fn visit_fru_field_types(&mut self) {
        let fcx_tables = self.fcx.tables.borrow();
        debug_assert_eq!(fcx_tables.local_id_root, self.tables.local_id_root);
        let common_local_id_root = fcx_tables.local_id_root.unwrap();

        for (&local_id, ftys) in fcx_tables.fru_field_types().iter() {
            let hir_id = hir::HirId {
                owner: common_local_id_root.index,
                local_id,
            };
            let ftys = self.resolve(ftys, &hir_id);
            self.tables.fru_field_types_mut().insert(hir_id, ftys);
        }
    }

    fn resolve<T>(&self, x: &T, span: &dyn Locatable) -> T::Lifted
    where
        T: TypeFoldable<'tcx> + ty::Lift<'gcx>,
    {
        let Ok(x) = x.fold_with(&mut Resolver::new(self.fcx, span, self.body));
        if let Some(lifted) = self.tcx().lift_to_global(&x) {
            lifted
        } else {
            span_bug!(
                span.to_span(&self.fcx.tcx),
                "writeback: `{:?}` missing from the global type context",
                x
            );
        }
    }
}

trait Locatable {
    fn to_span(&self, tcx: &TyCtxt<'_, '_, '_>) -> Span;
}

impl Locatable for Span {
    fn to_span(&self, _: &TyCtxt<'_, '_, '_>) -> Span {
        *self
    }
}

impl Locatable for ast::NodeId {
    fn to_span(&self, tcx: &TyCtxt<'_, '_, '_>) -> Span {
        tcx.hir().span(*self)
    }
}

impl Locatable for DefIndex {
    fn to_span(&self, tcx: &TyCtxt<'_, '_, '_>) -> Span {
        let hir_id = tcx.hir().def_index_to_hir_id(*self);
        tcx.hir().span_by_hir_id(hir_id)
    }
}

impl Locatable for hir::HirId {
    fn to_span(&self, tcx: &TyCtxt<'_, '_, '_>) -> Span {
        tcx.hir().span_by_hir_id(*self)
    }
}

///////////////////////////////////////////////////////////////////////////
// The Resolver. This is the type folding engine that detects
// unresolved types and so forth.

struct Resolver<'cx, 'gcx: 'cx + 'tcx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'gcx, 'tcx>,
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
    span: &'cx dyn Locatable,
    body: &'gcx hir::Body,
}

impl<'cx, 'gcx, 'tcx> Resolver<'cx, 'gcx, 'tcx> {
    fn new(
        fcx: &'cx FnCtxt<'cx, 'gcx, 'tcx>,
        span: &'cx dyn Locatable,
        body: &'gcx hir::Body,
    ) -> Resolver<'cx, 'gcx, 'tcx> {
        Resolver {
            tcx: fcx.tcx,
            infcx: fcx,
            span,
            body,
        }
    }

    fn report_error(&self, t: Ty<'tcx>) {
        if !self.tcx.sess.has_errors() {
            self.infcx
                .need_type_info_err(Some(self.body.id()), self.span.to_span(&self.tcx), t)
                .emit();
        }
    }
}

impl<'cx, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for Resolver<'cx, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'a>(&'a self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        match self.infcx.fully_resolve(&t) {
            Ok(t) => Ok(t),
            Err(_) => {
                debug!(
                    "Resolver::fold_ty: input type `{:?}` not fully resolvable",
                    t
                );
                self.report_error(t);
                Ok(self.tcx().types.err)
            }
        }
    }

    // FIXME This should be carefully checked
    // We could use `self.report_error` but it doesn't accept a ty::Region, right now.
    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        Ok(self.infcx.fully_resolve(&r).unwrap_or(self.tcx.types.re_static))
    }
}

///////////////////////////////////////////////////////////////////////////
// During type check, we store promises with the result of trait
// lookup rather than the actual results (because the results are not
// necessarily available immediately). These routines unwind the
// promises. It is expected that we will have already reported any
// errors that may be encountered, so if the promises store an error,
// a dummy result is returned.
