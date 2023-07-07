//! Borrow checker diagnostics.

use crate::session_diagnostics::{
    CaptureArgLabel, CaptureReasonLabel, CaptureReasonNote, CaptureReasonSuggest, CaptureVarCause,
    CaptureVarKind, CaptureVarPathUseCause, OnClosureNote,
};
use itertools::Itertools;
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, Namespace};
use rustc_hir::GeneratorKind;
use rustc_index::IndexSlice;
use rustc_infer::infer::LateBoundRegionConversionTime;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::{
    AggregateKind, CallSource, Constant, FakeReadCause, Local, LocalInfo, LocalKind, Location,
    Operand, Place, PlaceRef, ProjectionElem, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::print::Print;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_middle::util::{call_kind, CallDesugaringKind};
use rustc_mir_dataflow::move_paths::{InitLocation, LookupResult};
use rustc_span::def_id::LocalDefId;
use rustc_span::{symbol::sym, Span, Symbol, DUMMY_SP};
use rustc_target::abi::{FieldIdx, VariantIdx};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{
    type_known_to_meet_bound_modulo_regions, Obligation, ObligationCause,
};

use super::borrow_set::BorrowData;
use super::MirBorrowckCtxt;

mod find_all_local_uses;
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

pub(crate) use bound_region_errors::{ToUniverseInfo, UniverseInfo};
pub(crate) use mutability_errors::AccessKind;
pub(crate) use outlives_suggestion::OutlivesSuggestionBuilder;
pub(crate) use region_errors::{ErrorConstraintInfo, RegionErrorKind, RegionErrors};
pub(crate) use region_name::{RegionName, RegionNameSource};
pub(crate) use rustc_middle::util::CallKind;

pub(super) struct DescribePlaceOpt {
    pub including_downcast: bool,

    /// Enable/Disable tuple fields.
    /// For example `x` tuple. if it's `true` `x.0`. Otherwise `x`
    pub including_tuple_field: bool,
}

pub(super) struct IncludingTupleField(pub(super) bool);

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
        diag: &mut Diagnostic,
    ) -> bool {
        debug!("add_moved_or_invoked_closure_note: location={:?} place={:?}", location, place);
        let mut target = place.local_or_deref_local();
        for stmt in &self.body[location.block].statements[location.statement_index..] {
            debug!("add_moved_or_invoked_closure_note: stmt={:?} target={:?}", stmt, target);
            if let StatementKind::Assign(box (into, Rvalue::Use(from))) = &stmt.kind {
                debug!("add_fnonce_closure_note: into={:?} from={:?}", into, from);
                match from {
                    Operand::Copy(place) | Operand::Move(place)
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
                if Some(self.infcx.tcx.parent(id)) == self.infcx.tcx.lang_items().fn_once_trait() {
                    let closure = match args.first() {
                        Some(Operand::Copy(place) | Operand::Move(place))
                            if target == place.local_or_deref_local() =>
                        {
                            place.local_or_deref_local().unwrap()
                        }
                        _ => return false,
                    };

                    debug!("add_moved_or_invoked_closure_note: closure={:?}", closure);
                    if let ty::Closure(did, _) = self.body.local_decls[closure].ty.kind() {
                        let did = did.expect_local();
                        if let Some((span, hir_place)) = self.infcx.tcx.closure_kind_origin(did) {
                            diag.eager_subdiagnostic(
                                &self.infcx.tcx.sess.parse_sess.span_diagnostic,
                                OnClosureNote::InvokedTwice {
                                    place_name: &ty::place_to_string_for_capture(
                                        self.infcx.tcx,
                                        hir_place,
                                    ),
                                    span: *span,
                                },
                            );
                            return true;
                        }
                    }
                }
            }
        }

        // Check if we are just moving a closure after it has been invoked.
        if let Some(target) = target {
            if let ty::Closure(did, _) = self.body.local_decls[target].ty.kind() {
                let did = did.expect_local();
                if let Some((span, hir_place)) = self.infcx.tcx.closure_kind_origin(did) {
                    diag.eager_subdiagnostic(
                        &self.infcx.tcx.sess.parse_sess.span_diagnostic,
                        OnClosureNote::MovedTwice {
                            place_name: &ty::place_to_string_for_capture(self.infcx.tcx, hir_place),
                            span: *span,
                        },
                    );
                    return true;
                }
            }
        }
        false
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
    /// If the place is a temporary for instance, `None` will be returned.
    pub(super) fn describe_place(&self, place_ref: PlaceRef<'tcx>) -> Option<String> {
        self.describe_place_with_options(
            place_ref,
            DescribePlaceOpt { including_downcast: false, including_tuple_field: true },
        )
    }

    /// End-user visible description of `place` if one can be found. If the place is a temporary
    /// for instance, `None` will be returned.
    /// `IncludingDowncast` parameter makes the function return `None` if `ProjectionElem` is
    /// `Downcast` and `IncludingDowncast` is true
    pub(super) fn describe_place_with_options(
        &self,
        place: PlaceRef<'tcx>,
        opt: DescribePlaceOpt,
    ) -> Option<String> {
        let local = place.local;
        let mut autoderef_index = None;
        let mut buf = String::new();
        let mut ok = self.append_local_to_string(local, &mut buf);

        for (index, elem) in place.projection.into_iter().enumerate() {
            match elem {
                ProjectionElem::Deref => {
                    if index == 0 {
                        if self.body.local_decls[local].is_ref_for_guard() {
                            continue;
                        }
                        if let LocalInfo::StaticRef { def_id, .. } =
                            *self.body.local_decls[local].local_info()
                        {
                            buf.push_str(self.infcx.tcx.item_name(def_id).as_str());
                            ok = Ok(());
                            continue;
                        }
                    }
                    if let Some(field) = self.is_upvar_field_projection(PlaceRef {
                        local,
                        projection: place.projection.split_at(index + 1).0,
                    }) {
                        let var_index = field.index();
                        buf = self.upvars[var_index].place.to_string(self.infcx.tcx);
                        ok = Ok(());
                        if !self.upvars[var_index].by_ref {
                            buf.insert(0, '*');
                        }
                    } else {
                        if autoderef_index.is_none() {
                            autoderef_index =
                                match place.projection.into_iter().rev().find_position(|elem| {
                                    !matches!(
                                        elem,
                                        ProjectionElem::Deref | ProjectionElem::Downcast(..)
                                    )
                                }) {
                                    Some((index, _)) => Some(place.projection.len() - index),
                                    None => Some(0),
                                };
                        }
                        if index >= autoderef_index.unwrap() {
                            buf.insert(0, '*');
                        }
                    }
                }
                ProjectionElem::Downcast(..) if opt.including_downcast => return None,
                ProjectionElem::Downcast(..) => (),
                ProjectionElem::OpaqueCast(..) => (),
                ProjectionElem::Field(field, _ty) => {
                    // FIXME(project-rfc_2229#36): print capture precisely here.
                    if let Some(field) = self.is_upvar_field_projection(PlaceRef {
                        local,
                        projection: place.projection.split_at(index + 1).0,
                    }) {
                        buf = self.upvars[field.index()].place.to_string(self.infcx.tcx);
                        ok = Ok(());
                    } else {
                        let field_name = self.describe_field(
                            PlaceRef { local, projection: place.projection.split_at(index).0 },
                            *field,
                            IncludingTupleField(opt.including_tuple_field),
                        );
                        if let Some(field_name_str) = field_name {
                            buf.push('.');
                            buf.push_str(&field_name_str);
                        }
                    }
                }
                ProjectionElem::Index(index) => {
                    buf.push('[');
                    if self.append_local_to_string(*index, &mut buf).is_err() {
                        buf.push('_');
                    }
                    buf.push(']');
                }
                ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                    // Since it isn't possible to borrow an element on a particular index and
                    // then use another while the borrow is held, don't output indices details
                    // to avoid confusing the end-user
                    buf.push_str("[..]");
                }
            }
        }
        ok.ok().map(|_| buf)
    }

    fn describe_name(&self, place: PlaceRef<'tcx>) -> Option<Symbol> {
        for elem in place.projection.into_iter() {
            match elem {
                ProjectionElem::Downcast(Some(name), _) => {
                    return Some(*name);
                }
                _ => {}
            }
        }
        None
    }

    /// Appends end-user visible description of the `local` place to `buf`. If `local` doesn't have
    /// a name, or its name was generated by the compiler, then `Err` is returned
    fn append_local_to_string(&self, local: Local, buf: &mut String) -> Result<(), ()> {
        let decl = &self.body.local_decls[local];
        match self.local_names[local] {
            Some(name) if !decl.from_compiler_desugaring() => {
                buf.push_str(name.as_str());
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// End-user visible description of the `field`nth field of `base`
    fn describe_field(
        &self,
        place: PlaceRef<'tcx>,
        field: FieldIdx,
        including_tuple_field: IncludingTupleField,
    ) -> Option<String> {
        let place_ty = match place {
            PlaceRef { local, projection: [] } => PlaceTy::from_ty(self.body.local_decls[local].ty),
            PlaceRef { local, projection: [proj_base @ .., elem] } => match elem {
                ProjectionElem::Deref
                | ProjectionElem::Index(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {
                    PlaceRef { local, projection: proj_base }.ty(self.body, self.infcx.tcx)
                }
                ProjectionElem::Downcast(..) => place.ty(self.body, self.infcx.tcx),
                ProjectionElem::OpaqueCast(ty) => PlaceTy::from_ty(*ty),
                ProjectionElem::Field(_, field_type) => PlaceTy::from_ty(*field_type),
            },
        };
        self.describe_field_from_ty(
            place_ty.ty,
            field,
            place_ty.variant_index,
            including_tuple_field,
        )
    }

    /// End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(
        &self,
        ty: Ty<'_>,
        field: FieldIdx,
        variant_index: Option<VariantIdx>,
        including_tuple_field: IncludingTupleField,
    ) -> Option<String> {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(ty.boxed_ty(), field, variant_index, including_tuple_field)
        } else {
            match *ty.kind() {
                ty::Adt(def, _) => {
                    let variant = if let Some(idx) = variant_index {
                        assert!(def.is_enum());
                        &def.variant(idx)
                    } else {
                        def.non_enum_variant()
                    };
                    if !including_tuple_field.0 && variant.ctor_kind() == Some(CtorKind::Fn) {
                        return None;
                    }
                    Some(variant.fields[field].name.to_string())
                }
                ty::Tuple(_) => Some(field.index().to_string()),
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    self.describe_field_from_ty(ty, field, variant_index, including_tuple_field)
                }
                ty::Array(ty, _) | ty::Slice(ty) => {
                    self.describe_field_from_ty(ty, field, variant_index, including_tuple_field)
                }
                ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                    // We won't be borrowck'ing here if the closure came from another crate,
                    // so it's safe to call `expect_local`.
                    //
                    // We know the field exists so it's safe to call operator[] and `unwrap` here.
                    let def_id = def_id.expect_local();
                    let var_id =
                        self.infcx.tcx.closure_captures(def_id)[field.index()].get_root_variable();

                    Some(self.infcx.tcx.hir().name(var_id).to_string())
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!("End-user description not implemented for field access on `{:?}`", ty);
                }
            }
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
                    let InitLocation::Statement(loc) = init.location else { continue };

                    let bbd = &self.body[loc.block];
                    let is_terminator = bbd.statements.len() == loc.statement_index;
                    debug!(
                        "borrowed_content_source: loc={:?} is_terminator={:?}",
                        loc, is_terminator,
                    );
                    if !is_terminator {
                        continue;
                    } else if let Some(Terminator {
                        kind:
                            TerminatorKind::Call {
                                func,
                                call_source: CallSource::OverloadedOperator,
                                ..
                            },
                        ..
                    }) = &bbd.terminator
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

    /// Return the name of the provided `Ty` (that must be a reference) with a synthesized lifetime
    /// name where required.
    pub(super) fn get_name_for_ty(&self, ty: Ty<'tcx>, counter: usize) -> String {
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, Namespace::TypeNS);

        // We need to add synthesized lifetimes where appropriate. We do
        // this by hooking into the pretty printer and telling it to label the
        // lifetimes without names with the value `'0`.
        if let ty::Ref(region, ..) = ty.kind() {
            match **region {
                ty::ReLateBound(_, ty::BoundRegion { kind: br, .. })
                | ty::RePlaceholder(ty::PlaceholderRegion {
                    bound: ty::BoundRegion { kind: br, .. },
                    ..
                }) => printer.region_highlight_mode.highlighting_bound_region(br, counter),
                _ => {}
            }
        }

        ty.print(printer).unwrap().into_buffer()
    }

    /// Returns the name of the provided `Ty` (that must be a reference)'s region with a
    /// synthesized lifetime name where required.
    pub(super) fn get_region_name_for_ty(&self, ty: Ty<'tcx>, counter: usize) -> String {
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, Namespace::TypeNS);

        let region = if let ty::Ref(region, ..) = ty.kind() {
            match **region {
                ty::ReLateBound(_, ty::BoundRegion { kind: br, .. })
                | ty::RePlaceholder(ty::PlaceholderRegion {
                    bound: ty::BoundRegion { kind: br, .. },
                    ..
                }) => printer.region_highlight_mode.highlighting_bound_region(br, counter),
                _ => {}
            }
            region
        } else {
            bug!("ty for annotation of borrow region is not a reference");
        };

        region.print(printer).unwrap().into_buffer()
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
        kind: CallKind<'tcx>,
    },
    /// This access is caused by a `match` or `if let` pattern.
    PatUse(Span),
    /// This access has a single span associated to it: common case.
    OtherUse(Span),
}

impl UseSpans<'_> {
    pub(super) fn args_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { args_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse { fn_call_span, kind: CallKind::DerefCoercion { .. }, .. } => {
                fn_call_span
            }
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    /// Returns the span of `self`, in the case of a `ClosureUse` returns the `path_span`
    pub(super) fn var_or_use_path_span(self) -> Span {
        match self {
            UseSpans::ClosureUse { path_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse { fn_call_span, kind: CallKind::DerefCoercion { .. }, .. } => {
                fn_call_span
            }
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    /// Returns the span of `self`, in the case of a `ClosureUse` returns the `capture_kind_span`
    pub(super) fn var_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { capture_kind_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse { fn_call_span, kind: CallKind::DerefCoercion { .. }, .. } => {
                fn_call_span
            }
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    pub(super) fn generator_kind(self) -> Option<GeneratorKind> {
        match self {
            UseSpans::ClosureUse { generator_kind, .. } => generator_kind,
            _ => None,
        }
    }

    /// Add a span label to the arguments of the closure, if it exists.
    pub(super) fn args_subdiag(
        self,
        err: &mut Diagnostic,
        f: impl FnOnce(Span) -> CaptureArgLabel,
    ) {
        if let UseSpans::ClosureUse { args_span, .. } = self {
            err.subdiagnostic(f(args_span));
        }
    }

    /// Add a span label to the use of the captured variable, if it exists.
    /// only adds label to the `path_span`
    pub(super) fn var_path_only_subdiag(
        self,
        err: &mut Diagnostic,
        action: crate::InitializationRequiringAction,
    ) {
        use crate::InitializationRequiringAction::*;
        use CaptureVarPathUseCause::*;
        if let UseSpans::ClosureUse { generator_kind, path_span, .. } = self {
            match generator_kind {
                Some(_) => {
                    err.subdiagnostic(match action {
                        Borrow => BorrowInGenerator { path_span },
                        MatchOn | Use => UseInGenerator { path_span },
                        Assignment => AssignInGenerator { path_span },
                        PartialAssignment => AssignPartInGenerator { path_span },
                    });
                }
                None => {
                    err.subdiagnostic(match action {
                        Borrow => BorrowInClosure { path_span },
                        MatchOn | Use => UseInClosure { path_span },
                        Assignment => AssignInClosure { path_span },
                        PartialAssignment => AssignPartInClosure { path_span },
                    });
                }
            }
        }
    }

    /// Add a subdiagnostic to the use of the captured variable, if it exists.
    pub(super) fn var_subdiag(
        self,
        handler: Option<&rustc_errors::Handler>,
        err: &mut Diagnostic,
        kind: Option<rustc_middle::mir::BorrowKind>,
        f: impl FnOnce(Option<GeneratorKind>, Span) -> CaptureVarCause,
    ) {
        if let UseSpans::ClosureUse { generator_kind, capture_kind_span, path_span, .. } = self {
            if capture_kind_span != path_span {
                err.subdiagnostic(match kind {
                    Some(kd) => match kd {
                        rustc_middle::mir::BorrowKind::Shared
                        | rustc_middle::mir::BorrowKind::Shallow => {
                            CaptureVarKind::Immut { kind_span: capture_kind_span }
                        }

                        rustc_middle::mir::BorrowKind::Mut { .. } => {
                            CaptureVarKind::Mut { kind_span: capture_kind_span }
                        }
                    },
                    None => CaptureVarKind::Move { kind_span: capture_kind_span },
                });
            };
            let diag = f(generator_kind, path_span);
            match handler {
                Some(hd) => err.eager_subdiagnostic(hd, diag),
                None => err.subdiagnostic(diag),
            };
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

impl<'tcx> BorrowedContentSource<'tcx> {
    pub(super) fn describe_for_unnamed_place(&self, tcx: TyCtxt<'_>) -> String {
        match *self {
            BorrowedContentSource::DerefRawPointer => "a raw pointer".to_string(),
            BorrowedContentSource::DerefSharedRef => "a shared reference".to_string(),
            BorrowedContentSource::DerefMutableRef => "a mutable reference".to_string(),
            BorrowedContentSource::OverloadedDeref(ty) => ty
                .ty_adt_def()
                .and_then(|adt| match tcx.get_diagnostic_name(adt.did())? {
                    name @ (sym::Rc | sym::Arc) => Some(format!("an `{name}`")),
                    _ => None,
                })
                .unwrap_or_else(|| format!("dereference of `{ty}`")),
            BorrowedContentSource::OverloadedIndex(ty) => format!("index of `{ty}`"),
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
            BorrowedContentSource::OverloadedDeref(ty) => ty
                .ty_adt_def()
                .and_then(|adt| match tcx.get_diagnostic_name(adt.did())? {
                    name @ (sym::Rc | sym::Arc) => Some(format!("an `{name}`")),
                    _ => None,
                })
                .unwrap_or_else(|| format!("dereference of `{ty}`")),
            BorrowedContentSource::OverloadedIndex(ty) => format!("an index of `{ty}`"),
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

///helper struct for explain_captures()
struct CapturedMessageOpt {
    is_partial_move: bool,
    is_loop_message: bool,
    is_move_msg: bool,
    is_loop_move: bool,
    maybe_reinitialized_locations_is_empty: bool,
}

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Finds the spans associated to a move or copy of move_place at location.
    pub(super) fn move_spans(
        &self,
        moved_place: PlaceRef<'tcx>, // Could also be an upvar.
        location: Location,
    ) -> UseSpans<'tcx> {
        use self::UseSpans::*;

        let Some(stmt) = self.body[location.block].statements.get(location.statement_index) else {
            return OtherUse(self.body.source_info(location).span);
        };

        debug!("move_spans: moved_place={:?} location={:?} stmt={:?}", moved_place, location, stmt);
        if let StatementKind::Assign(box (_, Rvalue::Aggregate(kind, places))) = &stmt.kind
            && let AggregateKind::Closure(def_id, _) | AggregateKind::Generator(def_id, _, _) = **kind
        {
            debug!("move_spans: def_id={:?} places={:?}", def_id, places);
            let def_id = def_id.expect_local();
            if let Some((args_span, generator_kind, capture_kind_span, path_span)) =
                self.closure_span(def_id, moved_place, places)
            {
                return ClosureUse {
                    generator_kind,
                    args_span,
                    capture_kind_span,
                    path_span,
                };
            }
        }

        // StatementKind::FakeRead only contains a def_id if they are introduced as a result
        // of pattern matching within a closure.
        if let StatementKind::FakeRead(box (cause, place)) = stmt.kind {
            match cause {
                FakeReadCause::ForMatchedPlace(Some(closure_def_id))
                | FakeReadCause::ForLet(Some(closure_def_id)) => {
                    debug!("move_spans: def_id={:?} place={:?}", closure_def_id, place);
                    let places = &[Operand::Move(place)];
                    if let Some((args_span, generator_kind, capture_kind_span, path_span)) =
                        self.closure_span(closure_def_id, moved_place, IndexSlice::from_raw(places))
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
            kind: TerminatorKind::Call { fn_span, call_source, .. }, ..
        }) = &self.body[location.block].terminator
        {
            let Some((method_did, method_substs)) =
            rustc_middle::util::find_self_call(
                    self.infcx.tcx,
                    &self.body,
                    target_temp,
                    location.block,
                )
            else {
                return normal_ret;
            };

            let kind = call_kind(
                self.infcx.tcx,
                self.param_env,
                method_did,
                method_substs,
                *fn_span,
                call_source.from_hir_call(),
                Some(self.infcx.tcx.fn_arg_names(method_did)[0]),
            );

            return FnSelfUse {
                var_span: stmt.source_info.span,
                fn_call_span: *fn_span,
                fn_span: self.infcx.tcx.def_span(method_did),
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
            Some(Statement { kind: StatementKind::Assign(box (place, _)), .. }) => {
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

        // drop and replace might have moved the assignment to the next block
        let maybe_additional_statement =
            if let TerminatorKind::Drop { target: drop_target, .. } =
                self.body[location.block].terminator().kind
            {
                self.body[drop_target].statements.first()
            } else {
                None
            };

        let statements =
            self.body[location.block].statements[location.statement_index + 1..].iter();

        for stmt in statements.chain(maybe_additional_statement) {
            if let StatementKind::Assign(box (_, Rvalue::Aggregate(kind, places))) = &stmt.kind {
                let (&def_id, is_generator) = match kind {
                    box AggregateKind::Closure(def_id, _) => (def_id, false),
                    box AggregateKind::Generator(def_id, _, _) => (def_id, true),
                    _ => continue,
                };
                let def_id = def_id.expect_local();

                debug!(
                    "borrow_spans: def_id={:?} is_generator={:?} places={:?}",
                    def_id, is_generator, places
                );
                if let Some((args_span, generator_kind, capture_kind_span, path_span)) =
                    self.closure_span(def_id, Place::from(target).as_ref(), places)
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
        def_id: LocalDefId,
        target_place: PlaceRef<'tcx>,
        places: &IndexSlice<FieldIdx, Operand<'tcx>>,
    ) -> Option<(Span, Option<GeneratorKind>, Span, Span)> {
        debug!(
            "closure_span: def_id={:?} target_place={:?} places={:?}",
            def_id, target_place, places
        );
        let hir_id = self.infcx.tcx.hir().local_def_id_to_hir_id(def_id);
        let expr = &self.infcx.tcx.hir().expect_expr(hir_id).kind;
        debug!("closure_span: hir_id={:?} expr={:?}", hir_id, expr);
        if let hir::ExprKind::Closure(&hir::Closure { body, fn_decl_span, .. }) = expr {
            for (captured_place, place) in
                self.infcx.tcx.closure_captures(def_id).iter().zip(places)
            {
                match place {
                    Operand::Copy(place) | Operand::Move(place)
                        if target_place == place.as_ref() =>
                    {
                        debug!("closure_span: found captured local {:?}", place);
                        let body = self.infcx.tcx.hir().body(body);
                        let generator_kind = body.generator_kind();

                        return Some((
                            fn_decl_span,
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

    fn explain_captures(
        &mut self,
        err: &mut Diagnostic,
        span: Span,
        move_span: Span,
        move_spans: UseSpans<'tcx>,
        moved_place: Place<'tcx>,
        msg_opt: CapturedMessageOpt,
    ) {
        let CapturedMessageOpt {
            is_partial_move: is_partial,
            is_loop_message,
            is_move_msg,
            is_loop_move,
            maybe_reinitialized_locations_is_empty,
        } = msg_opt;
        if let UseSpans::FnSelfUse { var_span, fn_call_span, fn_span, kind } = move_spans {
            let place_name = self
                .describe_place(moved_place.as_ref())
                .map(|n| format!("`{n}`"))
                .unwrap_or_else(|| "value".to_owned());
            match kind {
                CallKind::FnCall { fn_trait_id, .. }
                    if Some(fn_trait_id) == self.infcx.tcx.lang_items().fn_once_trait() =>
                {
                    err.subdiagnostic(CaptureReasonLabel::Call {
                        fn_call_span,
                        place_name: &place_name,
                        is_partial,
                        is_loop_message,
                    });
                    err.subdiagnostic(CaptureReasonNote::FnOnceMoveInCall { var_span });
                }
                CallKind::Operator { self_arg, .. } => {
                    let self_arg = self_arg.unwrap();
                    err.subdiagnostic(CaptureReasonLabel::OperatorUse {
                        fn_call_span,
                        place_name: &place_name,
                        is_partial,
                        is_loop_message,
                    });
                    if self.fn_self_span_reported.insert(fn_span) {
                        err.subdiagnostic(CaptureReasonNote::LhsMoveByOperator {
                            span: self_arg.span,
                        });
                    }
                }
                CallKind::Normal { self_arg, desugaring, method_did, method_substs } => {
                    let self_arg = self_arg.unwrap();
                    let tcx = self.infcx.tcx;
                    if let Some((CallDesugaringKind::ForLoopIntoIter, _)) = desugaring {
                        let ty = moved_place.ty(self.body, tcx).ty;
                        let suggest = match tcx.get_diagnostic_item(sym::IntoIterator) {
                            Some(def_id) => type_known_to_meet_bound_modulo_regions(
                                &self.infcx,
                                self.param_env,
                                Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, ty),
                                def_id,
                            ),
                            _ => false,
                        };
                        if suggest {
                            err.subdiagnostic(CaptureReasonSuggest::IterateSlice {
                                ty,
                                span: move_span.shrink_to_lo(),
                            });
                        }

                        err.subdiagnostic(CaptureReasonLabel::ImplicitCall {
                            fn_call_span,
                            place_name: &place_name,
                            is_partial,
                            is_loop_message,
                        });
                        // If the moved place was a `&mut` ref, then we can
                        // suggest to reborrow it where it was moved, so it
                        // will still be valid by the time we get to the usage.
                        if let ty::Ref(_, _, hir::Mutability::Mut) =
                            moved_place.ty(self.body, self.infcx.tcx).ty.kind()
                        {
                            // If we are in a loop this will be suggested later.
                            if !is_loop_move {
                                err.span_suggestion_verbose(
                                    move_span.shrink_to_lo(),
                                    format!(
                                        "consider creating a fresh reborrow of {} here",
                                        self.describe_place(moved_place.as_ref())
                                            .map(|n| format!("`{n}`"))
                                            .unwrap_or_else(|| "the mutable reference".to_string()),
                                    ),
                                    "&mut *",
                                    Applicability::MachineApplicable,
                                );
                            }
                        }
                    } else {
                        if let Some((CallDesugaringKind::Await, _)) = desugaring {
                            err.subdiagnostic(CaptureReasonLabel::Await {
                                fn_call_span,
                                place_name: &place_name,
                                is_partial,
                                is_loop_message,
                            });
                        } else {
                            err.subdiagnostic(CaptureReasonLabel::MethodCall {
                                fn_call_span,
                                place_name: &place_name,
                                is_partial,
                                is_loop_message,
                            });
                        }
                        // Erase and shadow everything that could be passed to the new infcx.
                        let ty = moved_place.ty(self.body, tcx).ty;

                        if let ty::Adt(def, substs) = ty.kind()
                            && Some(def.did()) == tcx.lang_items().pin_type()
                            && let ty::Ref(_, _, hir::Mutability::Mut) = substs.type_at(0).kind()
                            && let self_ty = self.infcx.instantiate_binder_with_fresh_vars(
                                fn_call_span,
                                LateBoundRegionConversionTime::FnCall,
                                tcx.fn_sig(method_did).subst(tcx, method_substs).input(0),
                            )
                            && self.infcx.can_eq(self.param_env, ty, self_ty)
                        {
                            err.eager_subdiagnostic(
                                &self.infcx.tcx.sess.parse_sess.span_diagnostic,
                                CaptureReasonSuggest::FreshReborrow {
                                    span: fn_call_span.shrink_to_lo(),
                                });
                        }
                        if let Some(clone_trait) = tcx.lang_items().clone_trait()
                            && let trait_ref = ty::TraitRef::new(tcx, clone_trait, [ty])
                            && let o = Obligation::new(
                                tcx,
                                ObligationCause::dummy(),
                                self.param_env,
                                ty::Binder::dummy(trait_ref),
                            )
                            && self.infcx.predicate_must_hold_modulo_regions(&o)
                        {
                            err.span_suggestion_verbose(
                                fn_call_span.shrink_to_lo(),
                                "you can `clone` the value and consume it, but this might not be \
                                 your desired behavior",
                                "clone().".to_string(),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    // Avoid pointing to the same function in multiple different
                    // error messages.
                    if span != DUMMY_SP && self.fn_self_span_reported.insert(self_arg.span) {
                        self.explain_iterator_advancement_in_for_loop_if_applicable(
                            err,
                            span,
                            &move_spans,
                        );

                        let func = tcx.def_path_str(method_did);
                        err.subdiagnostic(CaptureReasonNote::FuncTakeSelf {
                            func,
                            place_name,
                            span: self_arg.span,
                        });
                    }
                    let parent_did = tcx.parent(method_did);
                    let parent_self_ty =
                        matches!(tcx.def_kind(parent_did), rustc_hir::def::DefKind::Impl { .. })
                            .then_some(parent_did)
                            .and_then(|did| match tcx.type_of(did).subst_identity().kind() {
                                ty::Adt(def, ..) => Some(def.did()),
                                _ => None,
                            });
                    let is_option_or_result = parent_self_ty.is_some_and(|def_id| {
                        matches!(tcx.get_diagnostic_name(def_id), Some(sym::Option | sym::Result))
                    });
                    if is_option_or_result && maybe_reinitialized_locations_is_empty {
                        err.subdiagnostic(CaptureReasonLabel::BorrowContent { var_span });
                    }
                }
                // Other desugarings takes &self, which cannot cause a move
                _ => {}
            }
        } else {
            if move_span != span || is_loop_message {
                err.subdiagnostic(CaptureReasonLabel::MovedHere {
                    move_span,
                    is_partial,
                    is_move_msg,
                    is_loop_message,
                });
            }
            // If the move error occurs due to a loop, don't show
            // another message for the same span
            if !is_loop_message {
                move_spans.var_subdiag(None, err, None, |kind, var_span| match kind {
                    Some(_) => CaptureVarCause::PartialMoveUseInGenerator { var_span, is_partial },
                    None => CaptureVarCause::PartialMoveUseInClosure { var_span, is_partial },
                })
            }
        }
    }
}
