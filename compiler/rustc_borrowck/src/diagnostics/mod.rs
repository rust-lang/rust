//! Borrow checker diagnostics.

use rustc_errors::DiagnosticBuilder;
use rustc_hir as hir;
use rustc_hir::def::Namespace;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItemGroup;
use rustc_hir::GeneratorKind;
use rustc_middle::mir::{
    AggregateKind, Constant, FakeReadCause, Field, Local, LocalInfo, LocalKind, Location, Operand,
    Place, PlaceRef, ProjectionElem, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::print::Print;
use rustc_middle::ty::{self, DefIdTree, Instance, Ty, TyCtxt};
use rustc_mir_dataflow::move_paths::{InitLocation, LookupResult};
use rustc_span::{
    hygiene::{DesugaringKind, ForLoopLoc},
    symbol::sym,
    Span,
};
use rustc_target::abi::VariantIdx;

use super::borrow_set::BorrowData;
use super::MirBorrowckCtxt;

mod find_use;
mod outlives_suggestion;
mod region_name;
mod var_name;

mod bound_region_errors;
mod conflict_errors;
mod explain_borrow;
mod move_errors;
mod mutability_errors;
mod region_errors;

crate use bound_region_errors::{ToUniverseInfo, UniverseInfo};
crate use mutability_errors::AccessKind;
crate use outlives_suggestion::OutlivesSuggestionBuilder;
crate use region_errors::{ErrorConstraintInfo, RegionErrorKind, RegionErrors};
crate use region_name::{RegionName, RegionNameSource};
use rustc_span::symbol::Ident;

pub(super) struct IncludingDowncast(pub(super) bool);

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Adds a suggestion when a closure is invoked twice with a moved variable or when a closure
    /// is moved after being invoked.
    ///
    /// ```text
    /// note: closure cannot be invoked more than once because it moves the variable `dict` out of
    ///       its environment
    ///   --> $DIR/issue-42065.rs:16:29
    ///    |
    /// LL |         for (key, value) in dict {
    ///    |                             ^^^^
    /// ```
    pub(super) fn add_moved_or_invoked_closure_note(
        &self,
        location: Location,
        place: PlaceRef<'tcx>,
        diag: &mut DiagnosticBuilder<'_>,
    ) {
        debug!("add_moved_or_invoked_closure_note: location={:?} place={:?}", location, place);
        let mut target = place.local_or_deref_local();
        for stmt in &self.body[location.block].statements[location.statement_index..] {
            debug!("add_moved_or_invoked_closure_note: stmt={:?} target={:?}", stmt, target);
            if let StatementKind::Assign(box (into, Rvalue::Use(from))) = &stmt.kind {
                debug!("add_fnonce_closure_note: into={:?} from={:?}", into, from);
                match from {
                    Operand::Copy(ref place) | Operand::Move(ref place)
                        if target == place.local_or_deref_local() =>
                    {
                        target = into.local_or_deref_local()
                    }
                    _ => {}
                }
            }
        }

        // Check if we are attempting to call a closure after it has been invoked.
        let terminator = self.body[location.block].terminator();
        debug!("add_moved_or_invoked_closure_note: terminator={:?}", terminator);
        if let TerminatorKind::Call {
            func: Operand::Constant(box Constant { literal, .. }),
            args,
            ..
        } = &terminator.kind
        {
            if let ty::FnDef(id, _) = *literal.ty().kind() {
                debug!("add_moved_or_invoked_closure_note: id={:?}", id);
                if self.infcx.tcx.parent(id) == self.infcx.tcx.lang_items().fn_once_trait() {
                    let closure = match args.first() {
                        Some(Operand::Copy(ref place)) | Some(Operand::Move(ref place))
                            if target == place.local_or_deref_local() =>
                        {
                            place.local_or_deref_local().unwrap()
                        }
                        _ => return,
                    };

                    debug!("add_moved_or_invoked_closure_note: closure={:?}", closure);
                    if let ty::Closure(did, _) = self.body.local_decls[closure].ty.kind() {
                        let did = did.expect_local();
                        let hir_id = self.infcx.tcx.hir().local_def_id_to_hir_id(did);

                        if let Some((span, hir_place)) =
                            self.infcx.tcx.typeck(did).closure_kind_origins().get(hir_id)
                        {
                            diag.span_note(
                                *span,
                                &format!(
                                    "closure cannot be invoked more than once because it moves the \
                                    variable `{}` out of its environment",
                                    ty::place_to_string_for_capture(self.infcx.tcx, hir_place)
                                ),
                            );
                            return;
                        }
                    }
                }
            }
        }

        // Check if we are just moving a closure after it has been invoked.
        if let Some(target) = target {
            if let ty::Closure(did, _) = self.body.local_decls[target].ty.kind() {
                let did = did.expect_local();
                let hir_id = self.infcx.tcx.hir().local_def_id_to_hir_id(did);

                if let Some((span, hir_place)) =
                    self.infcx.tcx.typeck(did).closure_kind_origins().get(hir_id)
                {
                    diag.span_note(
                        *span,
                        &format!(
                            "closure cannot be moved more than once as it is not `Copy` due to \
                             moving the variable `{}` out of its environment",
                            ty::place_to_string_for_capture(self.infcx.tcx, hir_place)
                        ),
                    );
                }
            }
        }
    }

    /// End-user visible description of `place` if one can be found.
    /// If the place is a temporary for instance, `"value"` will be returned.
    pub(super) fn describe_any_place(&self, place_ref: PlaceRef<'tcx>) -> String {
        match self.describe_place(place_ref) {
            Some(mut descr) => {
                // Surround descr with `backticks`.
                descr.reserve(2);
                descr.insert(0, '`');
                descr.push('`');
                descr
            }
            None => "value".to_string(),
        }
    }

    /// End-user visible description of `place` if one can be found.
    /// If the place is a temporary for instance, None will be returned.
    pub(super) fn describe_place(&self, place_ref: PlaceRef<'tcx>) -> Option<String> {
        self.describe_place_with_options(place_ref, IncludingDowncast(false))
    }

    /// End-user visible description of `place` if one can be found. If the
    /// place is a temporary for instance, None will be returned.
    /// `IncludingDowncast` parameter makes the function return `Err` if `ProjectionElem` is
    /// `Downcast` and `IncludingDowncast` is true
    pub(super) fn describe_place_with_options(
        &self,
        place: PlaceRef<'tcx>,
        including_downcast: IncludingDowncast,
    ) -> Option<String> {
        let mut buf = String::new();
        match self.append_place_to_string(place, &mut buf, false, &including_downcast) {
            Ok(()) => Some(buf),
            Err(()) => None,
        }
    }

    /// Appends end-user visible description of `place` to `buf`.
    fn append_place_to_string(
        &self,
        place: PlaceRef<'tcx>,
        buf: &mut String,
        mut autoderef: bool,
        including_downcast: &IncludingDowncast,
    ) -> Result<(), ()> {
        match place {
            PlaceRef { local, projection: [] } => {
                self.append_local_to_string(local, buf)?;
            }
            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_ref_for_guard() =>
            {
                self.append_place_to_string(
                    PlaceRef { local, projection: &[] },
                    buf,
                    autoderef,
                    &including_downcast,
                )?;
            }
            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_ref_to_static() =>
            {
                let local_info = &self.body.local_decls[local].local_info;
                if let Some(box LocalInfo::StaticRef { def_id, .. }) = *local_info {
                    buf.push_str(&self.infcx.tcx.item_name(def_id).as_str());
                } else {
                    unreachable!();
                }
            }
            PlaceRef { local, projection: [proj_base @ .., elem] } => {
                match elem {
                    ProjectionElem::Deref => {
                        let upvar_field_projection = self.is_upvar_field_projection(place);
                        if let Some(field) = upvar_field_projection {
                            let var_index = field.index();
                            let name = self.upvars[var_index].place.to_string(self.infcx.tcx);
                            if self.upvars[var_index].by_ref {
                                buf.push_str(&name);
                            } else {
                                buf.push('*');
                                buf.push_str(&name);
                            }
                        } else {
                            if autoderef {
                                // FIXME turn this recursion into iteration
                                self.append_place_to_string(
                                    PlaceRef { local, projection: proj_base },
                                    buf,
                                    autoderef,
                                    &including_downcast,
                                )?;
                            } else {
                                buf.push('*');
                                self.append_place_to_string(
                                    PlaceRef { local, projection: proj_base },
                                    buf,
                                    autoderef,
                                    &including_downcast,
                                )?;
                            }
                        }
                    }
                    ProjectionElem::Downcast(..) => {
                        self.append_place_to_string(
                            PlaceRef { local, projection: proj_base },
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        if including_downcast.0 {
                            return Err(());
                        }
                    }
                    ProjectionElem::Field(field, _ty) => {
                        autoderef = true;

                        // FIXME(project-rfc_2229#36): print capture precisely here.
                        let upvar_field_projection = self.is_upvar_field_projection(place);
                        if let Some(field) = upvar_field_projection {
                            let var_index = field.index();
                            let name = self.upvars[var_index].place.to_string(self.infcx.tcx);
                            buf.push_str(&name);
                        } else {
                            let field_name = self
                                .describe_field(PlaceRef { local, projection: proj_base }, *field);
                            self.append_place_to_string(
                                PlaceRef { local, projection: proj_base },
                                buf,
                                autoderef,
                                &including_downcast,
                            )?;
                            buf.push('.');
                            buf.push_str(&field_name);
                        }
                    }
                    ProjectionElem::Index(index) => {
                        autoderef = true;

                        self.append_place_to_string(
                            PlaceRef { local, projection: proj_base },
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        buf.push('[');
                        if self.append_local_to_string(*index, buf).is_err() {
                            buf.push('_');
                        }
                        buf.push(']');
                    }
                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                        autoderef = true;
                        // Since it isn't possible to borrow an element on a particular index and
                        // then use another while the borrow is held, don't output indices details
                        // to avoid confusing the end-user
                        self.append_place_to_string(
                            PlaceRef { local, projection: proj_base },
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        buf.push_str("[..]");
                    }
                };
            }
        }

        Ok(())
    }

    /// Appends end-user visible description of the `local` place to `buf`. If `local` doesn't have
    /// a name, or its name was generated by the compiler, then `Err` is returned
    fn append_local_to_string(&self, local: Local, buf: &mut String) -> Result<(), ()> {
        let decl = &self.body.local_decls[local];
        match self.local_names[local] {
            Some(name) if !decl.from_compiler_desugaring() => {
                buf.push_str(&name.as_str());
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// End-user visible description of the `field`nth field of `base`
    fn describe_field(&self, place: PlaceRef<'tcx>, field: Field) -> String {
        // FIXME Place2 Make this work iteratively
        match place {
            PlaceRef { local, projection: [] } => {
                let local = &self.body.local_decls[local];
                self.describe_field_from_ty(&local.ty, field, None)
            }
            PlaceRef { local, projection: [proj_base @ .., elem] } => match elem {
                ProjectionElem::Deref => {
                    self.describe_field(PlaceRef { local, projection: proj_base }, field)
                }
                ProjectionElem::Downcast(_, variant_index) => {
                    let base_ty = place.ty(self.body, self.infcx.tcx).ty;
                    self.describe_field_from_ty(&base_ty, field, Some(*variant_index))
                }
                ProjectionElem::Field(_, field_type) => {
                    self.describe_field_from_ty(&field_type, field, None)
                }
                ProjectionElem::Index(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {
                    self.describe_field(PlaceRef { local, projection: proj_base }, field)
                }
            },
        }
    }

    /// End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(
        &self,
        ty: Ty<'_>,
        field: Field,
        variant_index: Option<VariantIdx>,
    ) -> String {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(&ty.boxed_ty(), field, variant_index)
        } else {
            match *ty.kind() {
                ty::Adt(def, _) => {
                    let variant = if let Some(idx) = variant_index {
                        assert!(def.is_enum());
                        &def.variants[idx]
                    } else {
                        def.non_enum_variant()
                    };
                    variant.fields[field.index()].ident.to_string()
                }
                ty::Tuple(_) => field.index().to_string(),
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    self.describe_field_from_ty(&ty, field, variant_index)
                }
                ty::Array(ty, _) | ty::Slice(ty) => {
                    self.describe_field_from_ty(&ty, field, variant_index)
                }
                ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                    // We won't be borrowck'ing here if the closure came from another crate,
                    // so it's safe to call `expect_local`.
                    //
                    // We know the field exists so it's safe to call operator[] and `unwrap` here.
                    let var_id = self
                        .infcx
                        .tcx
                        .typeck(def_id.expect_local())
                        .closure_min_captures_flattened(def_id)
                        .nth(field.index())
                        .unwrap()
                        .get_root_variable();

                    self.infcx.tcx.hir().name(var_id).to_string()
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!("End-user description not implemented for field access on `{:?}`", ty);
                }
            }
        }
    }

    /// Add a note that a type does not implement `Copy`
    pub(super) fn note_type_does_not_implement_copy(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        place_desc: &str,
        ty: Ty<'tcx>,
        span: Option<Span>,
        move_prefix: &str,
    ) {
        let message = format!(
            "{}move occurs because {} has type `{}`, which does not implement the `Copy` trait",
            move_prefix, place_desc, ty,
        );
        if let Some(span) = span {
            err.span_label(span, message);
        } else {
            err.note(&message);
        }
    }

    pub(super) fn borrowed_content_source(
        &self,
        deref_base: PlaceRef<'tcx>,
    ) -> BorrowedContentSource<'tcx> {
        let tcx = self.infcx.tcx;

        // Look up the provided place and work out the move path index for it,
        // we'll use this to check whether it was originally from an overloaded
        // operator.
        match self.move_data.rev_lookup.find(deref_base) {
            LookupResult::Exact(mpi) | LookupResult::Parent(Some(mpi)) => {
                debug!("borrowed_content_source: mpi={:?}", mpi);

                for i in &self.move_data.init_path_map[mpi] {
                    let init = &self.move_data.inits[*i];
                    debug!("borrowed_content_source: init={:?}", init);
                    // We're only interested in statements that initialized a value, not the
                    // initializations from arguments.
                    let loc = match init.location {
                        InitLocation::Statement(stmt) => stmt,
                        _ => continue,
                    };

                    let bbd = &self.body[loc.block];
                    let is_terminator = bbd.statements.len() == loc.statement_index;
                    debug!(
                        "borrowed_content_source: loc={:?} is_terminator={:?}",
                        loc, is_terminator,
                    );
                    if !is_terminator {
                        continue;
                    } else if let Some(Terminator {
                        kind: TerminatorKind::Call { ref func, from_hir_call: false, .. },
                        ..
                    }) = bbd.terminator
                    {
                        if let Some(source) =
                            BorrowedContentSource::from_call(func.ty(self.body, tcx), tcx)
                        {
                            return source;
                        }
                    }
                }
            }
            // Base is a `static` so won't be from an overloaded operator
            _ => (),
        };

        // If we didn't find an overloaded deref or index, then assume it's a
        // built in deref and check the type of the base.
        let base_ty = deref_base.ty(self.body, tcx).ty;
        if base_ty.is_unsafe_ptr() {
            BorrowedContentSource::DerefRawPointer
        } else if base_ty.is_mutable_ptr() {
            BorrowedContentSource::DerefMutableRef
        } else {
            BorrowedContentSource::DerefSharedRef
        }
    }
}

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Return the name of the provided `Ty` (that must be a reference) with a synthesized lifetime
    /// name where required.
    pub(super) fn get_name_for_ty(&self, ty: Ty<'tcx>, counter: usize) -> String {
        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, &mut s, Namespace::TypeNS);

        // We need to add synthesized lifetimes where appropriate. We do
        // this by hooking into the pretty printer and telling it to label the
        // lifetimes without names with the value `'0`.
        match ty.kind() {
            ty::Ref(
                ty::RegionKind::ReLateBound(_, ty::BoundRegion { kind: br, .. })
                | ty::RegionKind::RePlaceholder(ty::PlaceholderRegion { name: br, .. }),
                _,
                _,
            ) => printer.region_highlight_mode.highlighting_bound_region(*br, counter),
            _ => {}
        }

        let _ = ty.print(printer);
        s
    }

    /// Returns the name of the provided `Ty` (that must be a reference)'s region with a
    /// synthesized lifetime name where required.
    pub(super) fn get_region_name_for_ty(&self, ty: Ty<'tcx>, counter: usize) -> String {
        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, &mut s, Namespace::TypeNS);

        let region = match ty.kind() {
            ty::Ref(region, _, _) => {
                match region {
                    ty::RegionKind::ReLateBound(_, ty::BoundRegion { kind: br, .. })
                    | ty::RegionKind::RePlaceholder(ty::PlaceholderRegion { name: br, .. }) => {
                        printer.region_highlight_mode.highlighting_bound_region(*br, counter)
                    }
                    _ => {}
                }

                region
            }
            _ => bug!("ty for annotation of borrow region is not a reference"),
        };

        let _ = region.print(printer);
        s
    }
}

/// The span(s) associated to a use of a place.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum UseSpans<'tcx> {
    /// The access is caused by capturing a variable for a closure.
    ClosureUse {
        /// This is true if the captured variable was from a generator.
        generator_kind: Option<GeneratorKind>,
        /// The span of the args of the closure, including the `move` keyword if
        /// it's present.
        args_span: Span,
        /// The span of the use resulting in capture kind
        /// Check `ty::CaptureInfo` for more details
        capture_kind_span: Span,
        /// The span of the use resulting in the captured path
        /// Check `ty::CaptureInfo` for more details
        path_span: Span,
    },
    /// The access is caused by using a variable as the receiver of a method
    /// that takes 'self'
    FnSelfUse {
        /// The span of the variable being moved
        var_span: Span,
        /// The span of the method call on the variable
        fn_call_span: Span,
        /// The definition span of the method being called
        fn_span: Span,
        kind: FnSelfUseKind<'tcx>,
    },
    /// This access is caused by a `match` or `if let` pattern.
    PatUse(Span),
    /// This access has a single span associated to it: common case.
    OtherUse(Span),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum FnSelfUseKind<'tcx> {
    /// A normal method call of the form `receiver.foo(a, b, c)`
    Normal {
        self_arg: Ident,
        implicit_into_iter: bool,
        /// Whether the self type of the method call has an `.as_ref()` method.
        /// Used for better diagnostics.
        is_option_or_result: bool,
    },
    /// A call to `FnOnce::call_once`, desugared from `my_closure(a, b, c)`
    FnOnceCall,
    /// A call to an operator trait, desuraged from operator syntax (e.g. `a << b`)
    Operator { self_arg: Ident },
    DerefCoercion {
        /// The `Span` of the `Target` associated type
        /// in the `Deref` impl we are using.
        deref_target: Span,
        /// The type `T::Deref` we are dereferencing to
        deref_target_ty: Ty<'tcx>,
    },
}

impl UseSpans<'_> {
    pub(super) fn args_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { args_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse {
                fn_call_span, kind: FnSelfUseKind::DerefCoercion { .. }, ..
            } => fn_call_span,
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    /// Returns the span of `self`, in the case of a `ClosureUse` returns the `path_span`
    pub(super) fn var_or_use_path_span(self) -> Span {
        match self {
            UseSpans::ClosureUse { path_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse {
                fn_call_span, kind: FnSelfUseKind::DerefCoercion { .. }, ..
            } => fn_call_span,
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    /// Returns the span of `self`, in the case of a `ClosureUse` returns the `capture_kind_span`
    pub(super) fn var_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { capture_kind_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse {
                fn_call_span, kind: FnSelfUseKind::DerefCoercion { .. }, ..
            } => fn_call_span,
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    pub(super) fn generator_kind(self) -> Option<GeneratorKind> {
        match self {
            UseSpans::ClosureUse { generator_kind, .. } => generator_kind,
            _ => None,
        }
    }

    // Add a span label to the arguments of the closure, if it exists.
    pub(super) fn args_span_label(
        self,
        err: &mut DiagnosticBuilder<'_>,
        message: impl Into<String>,
    ) {
        if let UseSpans::ClosureUse { args_span, .. } = self {
            err.span_label(args_span, message);
        }
    }

    // Add a span label to the use of the captured variable, if it exists.
    // only adds label to the `path_span`
    pub(super) fn var_span_label_path_only(
        self,
        err: &mut DiagnosticBuilder<'_>,
        message: impl Into<String>,
    ) {
        if let UseSpans::ClosureUse { path_span, .. } = self {
            err.span_label(path_span, message);
        }
    }

    // Add a span label to the use of the captured variable, if it exists.
    pub(super) fn var_span_label(
        self,
        err: &mut DiagnosticBuilder<'_>,
        message: impl Into<String>,
        kind_desc: impl Into<String>,
    ) {
        if let UseSpans::ClosureUse { capture_kind_span, path_span, .. } = self {
            if capture_kind_span == path_span {
                err.span_label(capture_kind_span, message);
            } else {
                let capture_kind_label =
                    format!("capture is {} because of use here", kind_desc.into());
                let path_label = message;
                err.span_label(capture_kind_span, capture_kind_label);
                err.span_label(path_span, path_label);
            }
        }
    }

    /// Returns `false` if this place is not used in a closure.
    pub(super) fn for_closure(&self) -> bool {
        match *self {
            UseSpans::ClosureUse { generator_kind, .. } => generator_kind.is_none(),
            _ => false,
        }
    }

    /// Returns `false` if this place is not used in a generator.
    pub(super) fn for_generator(&self) -> bool {
        match *self {
            UseSpans::ClosureUse { generator_kind, .. } => generator_kind.is_some(),
            _ => false,
        }
    }

    /// Describe the span associated with a use of a place.
    pub(super) fn describe(&self) -> String {
        match *self {
            UseSpans::ClosureUse { generator_kind, .. } => {
                if generator_kind.is_some() {
                    " in generator".to_string()
                } else {
                    " in closure".to_string()
                }
            }
            _ => String::new(),
        }
    }

    pub(super) fn or_else<F>(self, if_other: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            closure @ UseSpans::ClosureUse { .. } => closure,
            UseSpans::PatUse(_) | UseSpans::OtherUse(_) => if_other(),
            fn_self @ UseSpans::FnSelfUse { .. } => fn_self,
        }
    }
}

pub(super) enum BorrowedContentSource<'tcx> {
    DerefRawPointer,
    DerefMutableRef,
    DerefSharedRef,
    OverloadedDeref(Ty<'tcx>),
    OverloadedIndex(Ty<'tcx>),
}

impl BorrowedContentSource<'tcx> {
    pub(super) fn describe_for_unnamed_place(&self, tcx: TyCtxt<'_>) -> String {
        match *self {
            BorrowedContentSource::DerefRawPointer => "a raw pointer".to_string(),
            BorrowedContentSource::DerefSharedRef => "a shared reference".to_string(),
            BorrowedContentSource::DerefMutableRef => "a mutable reference".to_string(),
            BorrowedContentSource::OverloadedDeref(ty) => match ty.kind() {
                ty::Adt(def, _) if tcx.is_diagnostic_item(sym::Rc, def.did) => {
                    "an `Rc`".to_string()
                }
                ty::Adt(def, _) if tcx.is_diagnostic_item(sym::Arc, def.did) => {
                    "an `Arc`".to_string()
                }
                _ => format!("dereference of `{}`", ty),
            },
            BorrowedContentSource::OverloadedIndex(ty) => format!("index of `{}`", ty),
        }
    }

    pub(super) fn describe_for_named_place(&self) -> Option<&'static str> {
        match *self {
            BorrowedContentSource::DerefRawPointer => Some("raw pointer"),
            BorrowedContentSource::DerefSharedRef => Some("shared reference"),
            BorrowedContentSource::DerefMutableRef => Some("mutable reference"),
            // Overloaded deref and index operators should be evaluated into a
            // temporary. So we don't need a description here.
            BorrowedContentSource::OverloadedDeref(_)
            | BorrowedContentSource::OverloadedIndex(_) => None,
        }
    }

    pub(super) fn describe_for_immutable_place(&self, tcx: TyCtxt<'_>) -> String {
        match *self {
            BorrowedContentSource::DerefRawPointer => "a `*const` pointer".to_string(),
            BorrowedContentSource::DerefSharedRef => "a `&` reference".to_string(),
            BorrowedContentSource::DerefMutableRef => {
                bug!("describe_for_immutable_place: DerefMutableRef isn't immutable")
            }
            BorrowedContentSource::OverloadedDeref(ty) => match ty.kind() {
                ty::Adt(def, _) if tcx.is_diagnostic_item(sym::Rc, def.did) => {
                    "an `Rc`".to_string()
                }
                ty::Adt(def, _) if tcx.is_diagnostic_item(sym::Arc, def.did) => {
                    "an `Arc`".to_string()
                }
                _ => format!("a dereference of `{}`", ty),
            },
            BorrowedContentSource::OverloadedIndex(ty) => format!("an index of `{}`", ty),
        }
    }

    fn from_call(func: Ty<'tcx>, tcx: TyCtxt<'tcx>) -> Option<Self> {
        match *func.kind() {
            ty::FnDef(def_id, substs) => {
                let trait_id = tcx.trait_of_item(def_id)?;

                let lang_items = tcx.lang_items();
                if Some(trait_id) == lang_items.deref_trait()
                    || Some(trait_id) == lang_items.deref_mut_trait()
                {
                    Some(BorrowedContentSource::OverloadedDeref(substs.type_at(0)))
                } else if Some(trait_id) == lang_items.index_trait()
                    || Some(trait_id) == lang_items.index_mut_trait()
                {
                    Some(BorrowedContentSource::OverloadedIndex(substs.type_at(0)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Finds the spans associated to a move or copy of move_place at location.
    pub(super) fn move_spans(
        &self,
        moved_place: PlaceRef<'tcx>, // Could also be an upvar.
        location: Location,
    ) -> UseSpans<'tcx> {
        use self::UseSpans::*;

        let stmt = match self.body[location.block].statements.get(location.statement_index) {
            Some(stmt) => stmt,
            None => return OtherUse(self.body.source_info(location).span),
        };

        debug!("move_spans: moved_place={:?} location={:?} stmt={:?}", moved_place, location, stmt);
        if let StatementKind::Assign(box (_, Rvalue::Aggregate(ref kind, ref places))) = stmt.kind {
            match kind {
                box AggregateKind::Closure(def_id, _)
                | box AggregateKind::Generator(def_id, _, _) => {
                    debug!("move_spans: def_id={:?} places={:?}", def_id, places);
                    if let Some((args_span, generator_kind, capture_kind_span, path_span)) =
                        self.closure_span(*def_id, moved_place, places)
                    {
                        return ClosureUse {
                            generator_kind,
                            args_span,
                            capture_kind_span,
                            path_span,
                        };
                    }
                }
                _ => {}
            }
        }

        // StatementKind::FakeRead only contains a def_id if they are introduced as a result
        // of pattern matching within a closure.
        if let StatementKind::FakeRead(box (cause, ref place)) = stmt.kind {
            match cause {
                FakeReadCause::ForMatchedPlace(Some(closure_def_id))
                | FakeReadCause::ForLet(Some(closure_def_id)) => {
                    debug!("move_spans: def_id={:?} place={:?}", closure_def_id, place);
                    let places = &[Operand::Move(*place)];
                    if let Some((args_span, generator_kind, capture_kind_span, path_span)) =
                        self.closure_span(closure_def_id, moved_place, places)
                    {
                        return ClosureUse {
                            generator_kind,
                            args_span,
                            capture_kind_span,
                            path_span,
                        };
                    }
                }
                _ => {}
            }
        }

        let normal_ret =
            if moved_place.projection.iter().any(|p| matches!(p, ProjectionElem::Downcast(..))) {
                PatUse(stmt.source_info.span)
            } else {
                OtherUse(stmt.source_info.span)
            };

        // We are trying to find MIR of the form:
        // ```
        // _temp = _moved_val;
        // ...
        // FnSelfCall(_temp, ...)
        // ```
        //
        // where `_moved_val` is the place we generated the move error for,
        // `_temp` is some other local, and `FnSelfCall` is a function
        // that has a `self` parameter.

        let target_temp = match stmt.kind {
            StatementKind::Assign(box (temp, _)) if temp.as_local().is_some() => {
                temp.as_local().unwrap()
            }
            _ => return normal_ret,
        };

        debug!("move_spans: target_temp = {:?}", target_temp);

        if let Some(Terminator {
            kind: TerminatorKind::Call { fn_span, from_hir_call, .. }, ..
        }) = &self.body[location.block].terminator
        {
            let (method_did, method_substs) = if let Some(info) =
                rustc_const_eval::util::find_self_call(
                    self.infcx.tcx,
                    &self.body,
                    target_temp,
                    location.block,
                ) {
                info
            } else {
                return normal_ret;
            };

            let tcx = self.infcx.tcx;
            let parent = tcx.parent(method_did);
            let is_fn_once = parent == tcx.lang_items().fn_once_trait();
            let is_operator = !from_hir_call
                && parent.map_or(false, |p| tcx.lang_items().group(LangItemGroup::Op).contains(&p));
            let is_deref = !from_hir_call && tcx.is_diagnostic_item(sym::deref_method, method_did);
            let fn_call_span = *fn_span;

            let self_arg = tcx.fn_arg_names(method_did)[0];

            debug!(
                "terminator = {:?} from_hir_call={:?}",
                self.body[location.block].terminator, from_hir_call
            );

            // Check for a 'special' use of 'self' -
            // an FnOnce call, an operator (e.g. `<<`), or a
            // deref coercion.
            let kind = if is_fn_once {
                Some(FnSelfUseKind::FnOnceCall)
            } else if is_operator {
                Some(FnSelfUseKind::Operator { self_arg })
            } else if is_deref {
                let deref_target =
                    tcx.get_diagnostic_item(sym::deref_target).and_then(|deref_target| {
                        Instance::resolve(tcx, self.param_env, deref_target, method_substs)
                            .transpose()
                    });
                if let Some(Ok(instance)) = deref_target {
                    let deref_target_ty = instance.ty(tcx, self.param_env);
                    Some(FnSelfUseKind::DerefCoercion {
                        deref_target: tcx.def_span(instance.def_id()),
                        deref_target_ty,
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let kind = kind.unwrap_or_else(|| {
                // This isn't a 'special' use of `self`
                debug!("move_spans: method_did={:?}, fn_call_span={:?}", method_did, fn_call_span);
                let implicit_into_iter = matches!(
                    fn_call_span.desugaring_kind(),
                    Some(DesugaringKind::ForLoop(ForLoopLoc::IntoIter))
                );
                let parent_self_ty = parent
                    .filter(|did| tcx.def_kind(*did) == rustc_hir::def::DefKind::Impl)
                    .and_then(|did| match tcx.type_of(did).kind() {
                        ty::Adt(def, ..) => Some(def.did),
                        _ => None,
                    });
                let is_option_or_result = parent_self_ty.map_or(false, |def_id| {
                    tcx.is_diagnostic_item(sym::option_type, def_id)
                        || tcx.is_diagnostic_item(sym::result_type, def_id)
                });
                FnSelfUseKind::Normal { self_arg, implicit_into_iter, is_option_or_result }
            });

            return FnSelfUse {
                var_span: stmt.source_info.span,
                fn_call_span,
                fn_span: self
                    .infcx
                    .tcx
                    .sess
                    .source_map()
                    .guess_head_span(self.infcx.tcx.def_span(method_did)),
                kind,
            };
        }
        normal_ret
    }

    /// Finds the span of arguments of a closure (within `maybe_closure_span`)
    /// and its usage of the local assigned at `location`.
    /// This is done by searching in statements succeeding `location`
    /// and originating from `maybe_closure_span`.
    pub(super) fn borrow_spans(&self, use_span: Span, location: Location) -> UseSpans<'tcx> {
        use self::UseSpans::*;
        debug!("borrow_spans: use_span={:?} location={:?}", use_span, location);

        let target = match self.body[location.block].statements.get(location.statement_index) {
            Some(&Statement { kind: StatementKind::Assign(box (ref place, _)), .. }) => {
                if let Some(local) = place.as_local() {
                    local
                } else {
                    return OtherUse(use_span);
                }
            }
            _ => return OtherUse(use_span),
        };

        if self.body.local_kind(target) != LocalKind::Temp {
            // operands are always temporaries.
            return OtherUse(use_span);
        }

        for stmt in &self.body[location.block].statements[location.statement_index + 1..] {
            if let StatementKind::Assign(box (_, Rvalue::Aggregate(ref kind, ref places))) =
                stmt.kind
            {
                let (def_id, is_generator) = match kind {
                    box AggregateKind::Closure(def_id, _) => (def_id, false),
                    box AggregateKind::Generator(def_id, _, _) => (def_id, true),
                    _ => continue,
                };

                debug!(
                    "borrow_spans: def_id={:?} is_generator={:?} places={:?}",
                    def_id, is_generator, places
                );
                if let Some((args_span, generator_kind, capture_kind_span, path_span)) =
                    self.closure_span(*def_id, Place::from(target).as_ref(), places)
                {
                    return ClosureUse { generator_kind, args_span, capture_kind_span, path_span };
                } else {
                    return OtherUse(use_span);
                }
            }

            if use_span != stmt.source_info.span {
                break;
            }
        }

        OtherUse(use_span)
    }

    /// Finds the spans of a captured place within a closure or generator.
    /// The first span is the location of the use resulting in the capture kind of the capture
    /// The second span is the location the use resulting in the captured path of the capture
    fn closure_span(
        &self,
        def_id: DefId,
        target_place: PlaceRef<'tcx>,
        places: &[Operand<'tcx>],
    ) -> Option<(Span, Option<GeneratorKind>, Span, Span)> {
        debug!(
            "closure_span: def_id={:?} target_place={:?} places={:?}",
            def_id, target_place, places
        );
        let local_did = def_id.as_local()?;
        let hir_id = self.infcx.tcx.hir().local_def_id_to_hir_id(local_did);
        let expr = &self.infcx.tcx.hir().expect_expr(hir_id).kind;
        debug!("closure_span: hir_id={:?} expr={:?}", hir_id, expr);
        if let hir::ExprKind::Closure(.., body_id, args_span, _) = expr {
            for (captured_place, place) in self
                .infcx
                .tcx
                .typeck(def_id.expect_local())
                .closure_min_captures_flattened(def_id)
                .zip(places)
            {
                match place {
                    Operand::Copy(place) | Operand::Move(place)
                        if target_place == place.as_ref() =>
                    {
                        debug!("closure_span: found captured local {:?}", place);
                        let body = self.infcx.tcx.hir().body(*body_id);
                        let generator_kind = body.generator_kind();

                        return Some((
                            *args_span,
                            generator_kind,
                            captured_place.get_capture_kind_span(self.infcx.tcx),
                            captured_place.get_path_span(self.infcx.tcx),
                        ));
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Helper to retrieve span(s) of given borrow from the current MIR
    /// representation
    pub(super) fn retrieve_borrow_spans(&self, borrow: &BorrowData<'_>) -> UseSpans<'tcx> {
        let span = self.body.source_info(borrow.reserve_location).span;
        self.borrow_spans(span, borrow.reserve_location)
    }
}
