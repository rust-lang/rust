//! Borrow checker diagnostics.

use std::collections::BTreeMap;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::{Applicability, Diag, EmissionGuarantee, MultiSpan, listify};
use rustc_hir::def::{CtorKind, Namespace};
use rustc_hir::{self as hir, CoroutineKind, LangItem};
use rustc_index::IndexSlice;
use rustc_infer::infer::{BoundRegionConversionTime, NllRegionVariableOrigin};
use rustc_infer::traits::SelectionError;
use rustc_middle::bug;
use rustc_middle::mir::{
    AggregateKind, CallSource, ConstOperand, ConstraintCategory, FakeReadCause, Local, LocalInfo,
    LocalKind, Location, Operand, Place, PlaceRef, PlaceTy, ProjectionElem, Rvalue, Statement,
    StatementKind, Terminator, TerminatorKind, find_self_call,
};
use rustc_middle::ty::print::Print;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::move_paths::{InitLocation, LookupResult, MoveOutIndex};
use rustc_span::def_id::LocalDefId;
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, Symbol, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::traits::call_kind::{CallDesugaringKind, call_kind};
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::{
    FulfillmentError, FulfillmentErrorCode, type_known_to_meet_bound_modulo_regions,
};
use tracing::debug;

use super::MirBorrowckCtxt;
use super::borrow_set::BorrowData;
use crate::constraints::OutlivesConstraint;
use crate::fluent_generated as fluent;
use crate::nll::ConstraintDescription;
use crate::session_diagnostics::{
    CaptureArgLabel, CaptureReasonLabel, CaptureReasonNote, CaptureReasonSuggest, CaptureVarCause,
    CaptureVarKind, CaptureVarPathUseCause, OnClosureNote,
};

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
mod opaque_suggestions;
mod region_errors;

pub(crate) use bound_region_errors::{ToUniverseInfo, UniverseInfo};
pub(crate) use move_errors::{IllegalMoveOriginKind, MoveError};
pub(crate) use mutability_errors::AccessKind;
pub(crate) use outlives_suggestion::OutlivesSuggestionBuilder;
pub(crate) use region_errors::{ErrorConstraintInfo, RegionErrorKind, RegionErrors};
pub(crate) use region_name::{RegionName, RegionNameSource};
pub(crate) use rustc_trait_selection::error_reporting::traits::call_kind::CallKind;

pub(super) struct DescribePlaceOpt {
    including_downcast: bool,

    /// Enable/Disable tuple fields.
    /// For example `x` tuple. if it's `true` `x.0`. Otherwise `x`
    including_tuple_field: bool,
}

pub(super) struct IncludingTupleField(pub(super) bool);

enum BufferedDiag<'infcx> {
    Error(Diag<'infcx>),
    NonError(Diag<'infcx, ()>),
}

impl<'infcx> BufferedDiag<'infcx> {
    fn sort_span(&self) -> Span {
        match self {
            BufferedDiag::Error(diag) => diag.sort_span,
            BufferedDiag::NonError(diag) => diag.sort_span,
        }
    }
}

#[derive(Default)]
pub(crate) struct BorrowckDiagnosticsBuffer<'infcx, 'tcx> {
    /// This field keeps track of move errors that are to be reported for given move indices.
    ///
    /// There are situations where many errors can be reported for a single move out (see
    /// #53807) and we want only the best of those errors.
    ///
    /// The `report_use_of_moved_or_uninitialized` function checks this map and replaces the
    /// diagnostic (if there is one) if the `Place` of the error being reported is a prefix of
    /// the `Place` of the previous most diagnostic. This happens instead of buffering the
    /// error. Once all move errors have been reported, any diagnostics in this map are added
    /// to the buffer to be emitted.
    ///
    /// `BTreeMap` is used to preserve the order of insertions when iterating. This is necessary
    /// when errors in the map are being re-added to the error buffer so that errors with the
    /// same primary span come out in a consistent order.
    buffered_move_errors: BTreeMap<Vec<MoveOutIndex>, (PlaceRef<'tcx>, Diag<'infcx>)>,

    buffered_mut_errors: FxIndexMap<Span, (Diag<'infcx>, usize)>,

    /// Buffer of diagnostics to be reported. A mixture of error and non-error diagnostics.
    buffered_diags: Vec<BufferedDiag<'infcx>>,
}

impl<'infcx, 'tcx> BorrowckDiagnosticsBuffer<'infcx, 'tcx> {
    pub(crate) fn buffer_non_error(&mut self, diag: Diag<'infcx, ()>) {
        self.buffered_diags.push(BufferedDiag::NonError(diag));
    }
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
    pub(crate) fn buffer_error(&mut self, diag: Diag<'infcx>) {
        self.diags_buffer.buffered_diags.push(BufferedDiag::Error(diag));
    }

    pub(crate) fn buffer_non_error(&mut self, diag: Diag<'infcx, ()>) {
        self.diags_buffer.buffer_non_error(diag);
    }

    pub(crate) fn buffer_move_error(
        &mut self,
        move_out_indices: Vec<MoveOutIndex>,
        place_and_err: (PlaceRef<'tcx>, Diag<'infcx>),
    ) -> bool {
        if let Some((_, diag)) =
            self.diags_buffer.buffered_move_errors.insert(move_out_indices, place_and_err)
        {
            // Cancel the old diagnostic so we don't ICE
            diag.cancel();
            false
        } else {
            true
        }
    }

    pub(crate) fn get_buffered_mut_error(&mut self, span: Span) -> Option<(Diag<'infcx>, usize)> {
        // FIXME(#120456) - is `swap_remove` correct?
        self.diags_buffer.buffered_mut_errors.swap_remove(&span)
    }

    pub(crate) fn buffer_mut_error(&mut self, span: Span, diag: Diag<'infcx>, count: usize) {
        self.diags_buffer.buffered_mut_errors.insert(span, (diag, count));
    }

    pub(crate) fn emit_errors(&mut self) -> Option<ErrorGuaranteed> {
        let mut res = self.infcx.tainted_by_errors();

        // Buffer any move errors that we collected and de-duplicated.
        for (_, (_, diag)) in std::mem::take(&mut self.diags_buffer.buffered_move_errors) {
            // We have already set tainted for this error, so just buffer it.
            self.buffer_error(diag);
        }
        for (_, (mut diag, count)) in std::mem::take(&mut self.diags_buffer.buffered_mut_errors) {
            if count > 10 {
                #[allow(rustc::diagnostic_outside_of_impl)]
                #[allow(rustc::untranslatable_diagnostic)]
                diag.note(format!("...and {} other attempted mutable borrows", count - 10));
            }
            self.buffer_error(diag);
        }

        if !self.diags_buffer.buffered_diags.is_empty() {
            self.diags_buffer.buffered_diags.sort_by_key(|buffered_diag| buffered_diag.sort_span());
            for buffered_diag in self.diags_buffer.buffered_diags.drain(..) {
                match buffered_diag {
                    BufferedDiag::Error(diag) => res = Some(diag.emit()),
                    BufferedDiag::NonError(diag) => diag.emit(),
                }
            }
        }

        res
    }

    pub(crate) fn has_buffered_diags(&self) -> bool {
        self.diags_buffer.buffered_diags.is_empty()
    }

    pub(crate) fn has_move_error(
        &self,
        move_out_indices: &[MoveOutIndex],
    ) -> Option<&(PlaceRef<'tcx>, Diag<'infcx>)> {
        self.diags_buffer.buffered_move_errors.get(move_out_indices)
    }
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
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
    #[allow(rustc::diagnostic_outside_of_impl)] // FIXME
    pub(super) fn add_moved_or_invoked_closure_note(
        &self,
        location: Location,
        place: PlaceRef<'tcx>,
        diag: &mut Diag<'infcx>,
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
            func: Operand::Constant(box ConstOperand { const_, .. }),
            args,
            ..
        } = &terminator.kind
        {
            if let ty::FnDef(id, _) = *const_.ty().kind() {
                debug!("add_moved_or_invoked_closure_note: id={:?}", id);
                if self.infcx.tcx.is_lang_item(self.infcx.tcx.parent(id), LangItem::FnOnce) {
                    let closure = match args.first() {
                        Some(Spanned {
                            node: Operand::Copy(place) | Operand::Move(place), ..
                        }) if target == place.local_or_deref_local() => {
                            place.local_or_deref_local().unwrap()
                        }
                        _ => return false,
                    };

                    debug!("add_moved_or_invoked_closure_note: closure={:?}", closure);
                    if let ty::Closure(did, _) = self.body.local_decls[closure].ty.kind() {
                        let did = did.expect_local();
                        if let Some((span, hir_place)) = self.infcx.tcx.closure_kind_origin(did) {
                            diag.subdiagnostic(OnClosureNote::InvokedTwice {
                                place_name: &ty::place_to_string_for_capture(
                                    self.infcx.tcx,
                                    hir_place,
                                ),
                                span: *span,
                            });
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
                    diag.subdiagnostic(OnClosureNote::MovedTwice {
                        place_name: &ty::place_to_string_for_capture(self.infcx.tcx, hir_place),
                        span: *span,
                    });
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
                        buf = self.upvars[var_index].to_string(self.infcx.tcx);
                        ok = Ok(());
                        if !self.upvars[var_index].is_by_ref() {
                            buf.insert(0, '*');
                        }
                    } else {
                        if autoderef_index.is_none() {
                            autoderef_index = match place.projection.iter().rposition(|elem| {
                                !matches!(
                                    elem,
                                    ProjectionElem::Deref | ProjectionElem::Downcast(..)
                                )
                            }) {
                                Some(index) => Some(index + 1),
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
                ProjectionElem::Subtype(..) => (),
                ProjectionElem::UnwrapUnsafeBinder(_) => (),
                ProjectionElem::Field(field, _ty) => {
                    // FIXME(project-rfc_2229#36): print capture precisely here.
                    if let Some(field) = self.is_upvar_field_projection(PlaceRef {
                        local,
                        projection: place.projection.split_at(index + 1).0,
                    }) {
                        buf = self.upvars[field.index()].to_string(self.infcx.tcx);
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
                ProjectionElem::Subtype(ty)
                | ProjectionElem::OpaqueCast(ty)
                | ProjectionElem::UnwrapUnsafeBinder(ty) => PlaceTy::from_ty(*ty),
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
        if let Some(boxed_ty) = ty.boxed_ty() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(boxed_ty, field, variant_index, including_tuple_field)
        } else {
            match *ty.kind() {
                ty::Adt(def, _) => {
                    let variant = if let Some(idx) = variant_index {
                        assert!(def.is_enum());
                        def.variant(idx)
                    } else {
                        def.non_enum_variant()
                    };
                    if !including_tuple_field.0 && variant.ctor_kind() == Some(CtorKind::Fn) {
                        return None;
                    }
                    Some(variant.fields[field].name.to_string())
                }
                ty::Tuple(_) => Some(field.index().to_string()),
                ty::Ref(_, ty, _) | ty::RawPtr(ty, _) => {
                    self.describe_field_from_ty(ty, field, variant_index, including_tuple_field)
                }
                ty::Array(ty, _) | ty::Slice(ty) => {
                    self.describe_field_from_ty(ty, field, variant_index, including_tuple_field)
                }
                ty::Closure(def_id, _) | ty::Coroutine(def_id, _) => {
                    // We won't be borrowck'ing here if the closure came from another crate,
                    // so it's safe to call `expect_local`.
                    //
                    // We know the field exists so it's safe to call operator[] and `unwrap` here.
                    let def_id = def_id.expect_local();
                    let var_id =
                        self.infcx.tcx.closure_captures(def_id)[field.index()].get_root_variable();

                    Some(self.infcx.tcx.hir_name(var_id).to_string())
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
        if base_ty.is_raw_ptr() {
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
            match region.kind() {
                ty::ReBound(_, ty::BoundRegion { kind: br, .. })
                | ty::RePlaceholder(ty::PlaceholderRegion {
                    bound: ty::BoundRegion { kind: br, .. },
                    ..
                }) => printer.region_highlight_mode.highlighting_bound_region(br, counter),
                _ => {}
            }
        }

        ty.print(&mut printer).unwrap();
        printer.into_buffer()
    }

    /// Returns the name of the provided `Ty` (that must be a reference)'s region with a
    /// synthesized lifetime name where required.
    pub(super) fn get_region_name_for_ty(&self, ty: Ty<'tcx>, counter: usize) -> String {
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, Namespace::TypeNS);

        let region = if let ty::Ref(region, ..) = ty.kind() {
            match region.kind() {
                ty::ReBound(_, ty::BoundRegion { kind: br, .. })
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

        region.print(&mut printer).unwrap();
        printer.into_buffer()
    }

    /// Add a note to region errors and borrow explanations when higher-ranked regions in predicates
    /// implicitly introduce an "outlives `'static`" constraint.
    fn add_placeholder_from_predicate_note<G: EmissionGuarantee>(
        &self,
        err: &mut Diag<'_, G>,
        path: &[OutlivesConstraint<'tcx>],
    ) {
        let predicate_span = path.iter().find_map(|constraint| {
            let outlived = constraint.sub;
            if let Some(origin) = self.regioncx.definitions.get(outlived)
                && let NllRegionVariableOrigin::Placeholder(_) = origin.origin
                && let ConstraintCategory::Predicate(span) = constraint.category
            {
                Some(span)
            } else {
                None
            }
        });

        if let Some(span) = predicate_span {
            err.span_note(span, "due to current limitations in the borrow checker, this implies a `'static` lifetime");
        }
    }

    /// Add a label to region errors and borrow explanations when outlives constraints arise from
    /// proving a type implements `Sized` or `Copy`.
    fn add_sized_or_copy_bound_info<G: EmissionGuarantee>(
        &self,
        err: &mut Diag<'_, G>,
        blamed_category: ConstraintCategory<'tcx>,
        path: &[OutlivesConstraint<'tcx>],
    ) {
        for sought_category in [ConstraintCategory::SizedBound, ConstraintCategory::CopyBound] {
            if sought_category != blamed_category
                && let Some(sought_constraint) = path.iter().find(|c| c.category == sought_category)
            {
                let label = format!(
                    "requirement occurs due to {}",
                    sought_category.description().trim_end()
                );
                err.span_label(sought_constraint.span, label);
            }
        }
    }
}

/// The span(s) associated to a use of a place.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum UseSpans<'tcx> {
    /// The access is caused by capturing a variable for a closure.
    ClosureUse {
        /// This is true if the captured variable was from a coroutine.
        closure_kind: hir::ClosureKind,
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
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    /// Returns the span of `self`, in the case of a `ClosureUse` returns the `path_span`
    pub(super) fn var_or_use_path_span(self) -> Span {
        match self {
            UseSpans::ClosureUse { path_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    /// Returns the span of `self`, in the case of a `ClosureUse` returns the `capture_kind_span`
    pub(super) fn var_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { capture_kind_span: span, .. }
            | UseSpans::PatUse(span)
            | UseSpans::OtherUse(span) => span,
            UseSpans::FnSelfUse { var_span, .. } => var_span,
        }
    }

    // FIXME(coroutines): Make this just return the `ClosureKind` directly?
    pub(super) fn coroutine_kind(self) -> Option<CoroutineKind> {
        match self {
            UseSpans::ClosureUse {
                closure_kind: hir::ClosureKind::Coroutine(coroutine_kind),
                ..
            } => Some(coroutine_kind),
            _ => None,
        }
    }

    /// Add a span label to the arguments of the closure, if it exists.
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub(super) fn args_subdiag(self, err: &mut Diag<'_>, f: impl FnOnce(Span) -> CaptureArgLabel) {
        if let UseSpans::ClosureUse { args_span, .. } = self {
            err.subdiagnostic(f(args_span));
        }
    }

    /// Add a span label to the use of the captured variable, if it exists.
    /// only adds label to the `path_span`
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub(super) fn var_path_only_subdiag(
        self,
        err: &mut Diag<'_>,
        action: crate::InitializationRequiringAction,
    ) {
        use CaptureVarPathUseCause::*;

        use crate::InitializationRequiringAction::*;
        if let UseSpans::ClosureUse { closure_kind, path_span, .. } = self {
            match closure_kind {
                hir::ClosureKind::Coroutine(_) => {
                    err.subdiagnostic(match action {
                        Borrow => BorrowInCoroutine { path_span },
                        MatchOn | Use => UseInCoroutine { path_span },
                        Assignment => AssignInCoroutine { path_span },
                        PartialAssignment => AssignPartInCoroutine { path_span },
                    });
                }
                hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
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
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub(super) fn var_subdiag(
        self,
        err: &mut Diag<'_>,
        kind: Option<rustc_middle::mir::BorrowKind>,
        f: impl FnOnce(hir::ClosureKind, Span) -> CaptureVarCause,
    ) {
        if let UseSpans::ClosureUse { closure_kind, capture_kind_span, path_span, .. } = self {
            if capture_kind_span != path_span {
                err.subdiagnostic(match kind {
                    Some(kd) => match kd {
                        rustc_middle::mir::BorrowKind::Shared
                        | rustc_middle::mir::BorrowKind::Fake(_) => {
                            CaptureVarKind::Immut { kind_span: capture_kind_span }
                        }

                        rustc_middle::mir::BorrowKind::Mut { .. } => {
                            CaptureVarKind::Mut { kind_span: capture_kind_span }
                        }
                    },
                    None => CaptureVarKind::Move { kind_span: capture_kind_span },
                });
            };
            let diag = f(closure_kind, path_span);
            err.subdiagnostic(diag);
        }
    }

    /// Returns `false` if this place is not used in a closure.
    pub(super) fn for_closure(&self) -> bool {
        match *self {
            UseSpans::ClosureUse { closure_kind, .. } => {
                matches!(closure_kind, hir::ClosureKind::Closure)
            }
            _ => false,
        }
    }

    /// Returns `false` if this place is not used in a coroutine.
    pub(super) fn for_coroutine(&self) -> bool {
        match *self {
            // FIXME(coroutines): Do we want this to apply to synthetic coroutines?
            UseSpans::ClosureUse { closure_kind, .. } => {
                matches!(closure_kind, hir::ClosureKind::Coroutine(..))
            }
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
            ty::FnDef(def_id, args) => {
                let trait_id = tcx.trait_of_item(def_id)?;

                if tcx.is_lang_item(trait_id, LangItem::Deref)
                    || tcx.is_lang_item(trait_id, LangItem::DerefMut)
                {
                    Some(BorrowedContentSource::OverloadedDeref(args.type_at(0)))
                } else if tcx.is_lang_item(trait_id, LangItem::Index)
                    || tcx.is_lang_item(trait_id, LangItem::IndexMut)
                {
                    Some(BorrowedContentSource::OverloadedIndex(args.type_at(0)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Helper struct for `explain_captures`.
struct CapturedMessageOpt {
    is_partial_move: bool,
    is_loop_message: bool,
    is_move_msg: bool,
    is_loop_move: bool,
    has_suggest_reborrow: bool,
    maybe_reinitialized_locations_is_empty: bool,
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
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
            && let AggregateKind::Closure(def_id, _) | AggregateKind::Coroutine(def_id, _) = **kind
        {
            debug!("move_spans: def_id={:?} places={:?}", def_id, places);
            let def_id = def_id.expect_local();
            if let Some((args_span, closure_kind, capture_kind_span, path_span)) =
                self.closure_span(def_id, moved_place, places)
            {
                return ClosureUse { closure_kind, args_span, capture_kind_span, path_span };
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
                    if let Some((args_span, closure_kind, capture_kind_span, path_span)) =
                        self.closure_span(closure_def_id, moved_place, IndexSlice::from_raw(places))
                    {
                        return ClosureUse {
                            closure_kind,
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
            let Some((method_did, method_args)) =
                find_self_call(self.infcx.tcx, self.body, target_temp, location.block)
            else {
                return normal_ret;
            };

            let kind = call_kind(
                self.infcx.tcx,
                self.infcx.typing_env(self.infcx.param_env),
                method_did,
                method_args,
                *fn_span,
                call_source.from_hir_call(),
                self.infcx.tcx.fn_arg_idents(method_did)[0],
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
                let (&def_id, is_coroutine) = match kind {
                    box AggregateKind::Closure(def_id, _) => (def_id, false),
                    box AggregateKind::Coroutine(def_id, _) => (def_id, true),
                    _ => continue,
                };
                let def_id = def_id.expect_local();

                debug!(
                    "borrow_spans: def_id={:?} is_coroutine={:?} places={:?}",
                    def_id, is_coroutine, places
                );
                if let Some((args_span, closure_kind, capture_kind_span, path_span)) =
                    self.closure_span(def_id, Place::from(target).as_ref(), places)
                {
                    return ClosureUse { closure_kind, args_span, capture_kind_span, path_span };
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

    /// Finds the spans of a captured place within a closure or coroutine.
    /// The first span is the location of the use resulting in the capture kind of the capture
    /// The second span is the location the use resulting in the captured path of the capture
    fn closure_span(
        &self,
        def_id: LocalDefId,
        target_place: PlaceRef<'tcx>,
        places: &IndexSlice<FieldIdx, Operand<'tcx>>,
    ) -> Option<(Span, hir::ClosureKind, Span, Span)> {
        debug!(
            "closure_span: def_id={:?} target_place={:?} places={:?}",
            def_id, target_place, places
        );
        let hir_id = self.infcx.tcx.local_def_id_to_hir_id(def_id);
        let expr = &self.infcx.tcx.hir_expect_expr(hir_id).kind;
        debug!("closure_span: hir_id={:?} expr={:?}", hir_id, expr);
        if let &hir::ExprKind::Closure(&hir::Closure { kind, fn_decl_span, .. }) = expr {
            for (captured_place, place) in
                self.infcx.tcx.closure_captures(def_id).iter().zip(places)
            {
                match place {
                    Operand::Copy(place) | Operand::Move(place)
                        if target_place == place.as_ref() =>
                    {
                        debug!("closure_span: found captured local {:?}", place);
                        return Some((
                            fn_decl_span,
                            kind,
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

    #[allow(rustc::diagnostic_outside_of_impl)]
    #[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
    fn explain_captures(
        &mut self,
        err: &mut Diag<'infcx>,
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
            has_suggest_reborrow,
            maybe_reinitialized_locations_is_empty,
        } = msg_opt;
        if let UseSpans::FnSelfUse { var_span, fn_call_span, fn_span, kind } = move_spans {
            let place_name = self
                .describe_place(moved_place.as_ref())
                .map(|n| format!("`{n}`"))
                .unwrap_or_else(|| "value".to_owned());
            match kind {
                CallKind::FnCall { fn_trait_id, self_ty }
                    if self.infcx.tcx.is_lang_item(fn_trait_id, LangItem::FnOnce) =>
                {
                    err.subdiagnostic(CaptureReasonLabel::Call {
                        fn_call_span,
                        place_name: &place_name,
                        is_partial,
                        is_loop_message,
                    });
                    // Check if the move occurs on a value because of a call on a closure that comes
                    // from a type parameter `F: FnOnce()`. If so, we provide a targeted `note`:
                    // ```
                    // error[E0382]: use of moved value: `blk`
                    //   --> $DIR/once-cant-call-twice-on-heap.rs:8:5
                    //    |
                    // LL | fn foo<F:FnOnce()>(blk: F) {
                    //    |                    --- move occurs because `blk` has type `F`, which does not implement the `Copy` trait
                    // LL | blk();
                    //    | ----- `blk` moved due to this call
                    // LL | blk();
                    //    | ^^^ value used here after move
                    //    |
                    // note: `FnOnce` closures can only be called once
                    //   --> $DIR/once-cant-call-twice-on-heap.rs:6:10
                    //    |
                    // LL | fn foo<F:FnOnce()>(blk: F) {
                    //    |        ^^^^^^^^ `F` is made to be an `FnOnce` closure here
                    // LL | blk();
                    //    | ----- this value implements `FnOnce`, which causes it to be moved when called
                    // ```
                    if let ty::Param(param_ty) = *self_ty.kind()
                        && let generics = self.infcx.tcx.generics_of(self.mir_def_id())
                        && let param = generics.type_param(param_ty, self.infcx.tcx)
                        && let Some(hir_generics) = self
                            .infcx
                            .tcx
                            .typeck_root_def_id(self.mir_def_id().to_def_id())
                            .as_local()
                            .and_then(|def_id| self.infcx.tcx.hir_get_generics(def_id))
                        && let spans = hir_generics
                            .predicates
                            .iter()
                            .filter_map(|pred| match pred.kind {
                                hir::WherePredicateKind::BoundPredicate(pred) => Some(pred),
                                _ => None,
                            })
                            .filter(|pred| {
                                if let Some((id, _)) = pred.bounded_ty.as_generic_param() {
                                    id == param.def_id
                                } else {
                                    false
                                }
                            })
                            .flat_map(|pred| pred.bounds)
                            .filter_map(|bound| {
                                if let Some(trait_ref) = bound.trait_ref()
                                    && let Some(trait_def_id) = trait_ref.trait_def_id()
                                    && trait_def_id == fn_trait_id
                                {
                                    Some(bound.span())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<Span>>()
                        && !spans.is_empty()
                    {
                        let mut span: MultiSpan = spans.clone().into();
                        for sp in spans {
                            span.push_span_label(sp, fluent::borrowck_moved_a_fn_once_in_call_def);
                        }
                        span.push_span_label(
                            fn_call_span,
                            fluent::borrowck_moved_a_fn_once_in_call,
                        );
                        err.span_note(span, fluent::borrowck_moved_a_fn_once_in_call_call);
                    } else {
                        err.subdiagnostic(CaptureReasonNote::FnOnceMoveInCall { var_span });
                    }
                }
                CallKind::Operator { self_arg, trait_id, .. } => {
                    let self_arg = self_arg.unwrap();
                    err.subdiagnostic(CaptureReasonLabel::OperatorUse {
                        fn_call_span,
                        place_name: &place_name,
                        is_partial,
                        is_loop_message,
                    });
                    if self.fn_self_span_reported.insert(fn_span) {
                        let lang = self.infcx.tcx.lang_items();
                        err.subdiagnostic(
                            if [lang.not_trait(), lang.deref_trait(), lang.neg_trait()]
                                .contains(&Some(trait_id))
                            {
                                CaptureReasonNote::UnOpMoveByOperator { span: self_arg.span }
                            } else {
                                CaptureReasonNote::LhsMoveByOperator { span: self_arg.span }
                            },
                        );
                    }
                }
                CallKind::Normal { self_arg, desugaring, method_did, method_args } => {
                    let self_arg = self_arg.unwrap();
                    let mut has_sugg = false;
                    let tcx = self.infcx.tcx;
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
                            place_name: place_name.clone(),
                            span: self_arg.span,
                        });
                    }
                    let parent_did = tcx.parent(method_did);
                    let parent_self_ty =
                        matches!(tcx.def_kind(parent_did), rustc_hir::def::DefKind::Impl { .. })
                            .then_some(parent_did)
                            .and_then(|did| match tcx.type_of(did).instantiate_identity().kind() {
                                ty::Adt(def, ..) => Some(def.did()),
                                _ => None,
                            });
                    let is_option_or_result = parent_self_ty.is_some_and(|def_id| {
                        matches!(tcx.get_diagnostic_name(def_id), Some(sym::Option | sym::Result))
                    });
                    if is_option_or_result && maybe_reinitialized_locations_is_empty {
                        err.subdiagnostic(CaptureReasonLabel::BorrowContent { var_span });
                    }
                    if let Some((CallDesugaringKind::ForLoopIntoIter, _)) = desugaring {
                        let ty = moved_place.ty(self.body, tcx).ty;
                        let suggest = match tcx.get_diagnostic_item(sym::IntoIterator) {
                            Some(def_id) => type_known_to_meet_bound_modulo_regions(
                                self.infcx,
                                self.infcx.param_env,
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
                            // Suggest `reborrow` in other place for following situations:
                            // 1. If we are in a loop this will be suggested later.
                            // 2. If the moved value is a mut reference, it is used in a
                            // generic function and the corresponding arg's type is generic param.
                            if !is_loop_move && !has_suggest_reborrow {
                                self.suggest_reborrow(
                                    err,
                                    move_span.shrink_to_lo(),
                                    moved_place.as_ref(),
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

                        if let ty::Adt(def, args) = ty.peel_refs().kind()
                            && tcx.is_lang_item(def.did(), LangItem::Pin)
                            && let ty::Ref(_, _, hir::Mutability::Mut) = args.type_at(0).kind()
                            && let self_ty = self.infcx.instantiate_binder_with_fresh_vars(
                                fn_call_span,
                                BoundRegionConversionTime::FnCall,
                                tcx.fn_sig(method_did).instantiate(tcx, method_args).input(0),
                            )
                            && self.infcx.can_eq(self.infcx.param_env, ty, self_ty)
                        {
                            err.subdiagnostic(CaptureReasonSuggest::FreshReborrow {
                                span: move_span.shrink_to_hi(),
                            });
                            has_sugg = true;
                        }
                        if let Some(clone_trait) = tcx.lang_items().clone_trait() {
                            let sugg = if moved_place
                                .iter_projections()
                                .any(|(_, elem)| matches!(elem, ProjectionElem::Deref))
                            {
                                let (start, end) = if let Some(expr) = self.find_expr(move_span)
                                    && let Some(_) = self.clone_on_reference(expr)
                                    && let hir::ExprKind::MethodCall(_, rcvr, _, _) = expr.kind
                                {
                                    (move_span.shrink_to_lo(), move_span.with_lo(rcvr.span.hi()))
                                } else {
                                    (move_span.shrink_to_lo(), move_span.shrink_to_hi())
                                };
                                vec![
                                    // We use the fully-qualified path because `.clone()` can
                                    // sometimes choose `<&T as Clone>` instead of `<T as Clone>`
                                    // when going through auto-deref, so this ensures that doesn't
                                    // happen, causing suggestions for `.clone().clone()`.
                                    (start, format!("<{ty} as Clone>::clone(&")),
                                    (end, ")".to_string()),
                                ]
                            } else {
                                vec![(move_span.shrink_to_hi(), ".clone()".to_string())]
                            };
                            if let Some(errors) = self.infcx.type_implements_trait_shallow(
                                clone_trait,
                                ty,
                                self.infcx.param_env,
                            ) && !has_sugg
                            {
                                let msg = match &errors[..] {
                                    [] => "you can `clone` the value and consume it, but this \
                                           might not be your desired behavior"
                                        .to_string(),
                                    [error] => {
                                        format!(
                                            "you could `clone` the value and consume it, if the \
                                             `{}` trait bound could be satisfied",
                                            error.obligation.predicate,
                                        )
                                    }
                                    _ => {
                                        format!(
                                            "you could `clone` the value and consume it, if the \
                                             following trait bounds could be satisfied: {}",
                                            listify(&errors, |e: &FulfillmentError<'tcx>| format!(
                                                "`{}`",
                                                e.obligation.predicate
                                            ))
                                            .unwrap(),
                                        )
                                    }
                                };
                                err.multipart_suggestion_verbose(
                                    msg,
                                    sugg,
                                    Applicability::MaybeIncorrect,
                                );
                                for error in errors {
                                    if let FulfillmentErrorCode::Select(
                                        SelectionError::Unimplemented,
                                    ) = error.code
                                        && let ty::PredicateKind::Clause(ty::ClauseKind::Trait(
                                            pred,
                                        )) = error.obligation.predicate.kind().skip_binder()
                                    {
                                        self.infcx.err_ctxt().suggest_derive(
                                            &error.obligation,
                                            err,
                                            error.obligation.predicate.kind().rebind(pred),
                                        );
                                    }
                                }
                            }
                        }
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
                move_spans.var_subdiag(err, None, |kind, var_span| match kind {
                    hir::ClosureKind::Coroutine(_) => {
                        CaptureVarCause::PartialMoveUseInCoroutine { var_span, is_partial }
                    }
                    hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                        CaptureVarCause::PartialMoveUseInClosure { var_span, is_partial }
                    }
                })
            }
        }
    }
}
