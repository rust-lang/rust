//! Borrow checker diagnostics.

use rustc::mir::{
    AggregateKind, Constant, Field, Local, LocalInfo, LocalKind, Location, Operand, Place,
    PlaceRef, ProjectionElem, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc::ty::layout::VariantIdx;
use rustc::ty::print::Print;
use rustc::ty::{self, DefIdTree, Ty, TyCtxt};
use rustc_errors::DiagnosticBuilder;
use rustc_hir as hir;
use rustc_hir::def::Namespace;
use rustc_hir::def_id::DefId;
use rustc_hir::GeneratorKind;
use rustc_span::Span;

use super::borrow_set::BorrowData;
use super::MirBorrowckCtxt;
use crate::dataflow::move_paths::{InitLocation, LookupResult};

mod find_use;
mod outlives_suggestion;
mod region_name;
mod var_name;

mod conflict_errors;
mod explain_borrow;
mod move_errors;
mod mutability_errors;
mod region_errors;

crate use mutability_errors::AccessKind;
crate use outlives_suggestion::OutlivesSuggestionBuilder;
crate use region_errors::{ErrorConstraintInfo, RegionErrorKind, RegionErrors};
crate use region_name::{RegionName, RegionNameSource};

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
        place: PlaceRef<'cx, 'tcx>,
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
            func:
                Operand::Constant(box Constant {
                    literal: ty::Const { ty: &ty::TyS { kind: ty::FnDef(id, _), .. }, .. },
                    ..
                }),
            args,
            ..
        } = &terminator.kind
        {
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
                if let ty::Closure(did, _) = self.body.local_decls[closure].ty.kind {
                    let hir_id = self.infcx.tcx.hir().as_local_hir_id(did).unwrap();

                    if let Some((span, name)) =
                        self.infcx.tcx.typeck_tables_of(did).closure_kind_origins().get(hir_id)
                    {
                        diag.span_note(
                            *span,
                            &format!(
                                "closure cannot be invoked more than once because it moves the \
                                 variable `{}` out of its environment",
                                name,
                            ),
                        );
                        return;
                    }
                }
            }
        }

        // Check if we are just moving a closure after it has been invoked.
        if let Some(target) = target {
            if let ty::Closure(did, _) = self.body.local_decls[target].ty.kind {
                let hir_id = self.infcx.tcx.hir().as_local_hir_id(did).unwrap();

                if let Some((span, name)) =
                    self.infcx.tcx.typeck_tables_of(did).closure_kind_origins().get(hir_id)
                {
                    diag.span_note(
                        *span,
                        &format!(
                            "closure cannot be moved more than once as it is not `Copy` due to \
                             moving the variable `{}` out of its environment",
                            name
                        ),
                    );
                }
            }
        }
    }

    /// End-user visible description of `place` if one can be found. If the
    /// place is a temporary for instance, None will be returned.
    pub(super) fn describe_place(&self, place_ref: PlaceRef<'cx, 'tcx>) -> Option<String> {
        self.describe_place_with_options(place_ref, IncludingDowncast(false))
    }

    /// End-user visible description of `place` if one can be found. If the
    /// place is a temporary for instance, None will be returned.
    /// `IncludingDowncast` parameter makes the function return `Err` if `ProjectionElem` is
    /// `Downcast` and `IncludingDowncast` is true
    pub(super) fn describe_place_with_options(
        &self,
        place: PlaceRef<'cx, 'tcx>,
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
        place: PlaceRef<'cx, 'tcx>,
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
                    PlaceRef { local: local, projection: &[] },
                    buf,
                    autoderef,
                    &including_downcast,
                )?;
            }
            PlaceRef { local, projection: [ProjectionElem::Deref] }
                if self.body.local_decls[local].is_ref_to_static() =>
            {
                let local_info = &self.body.local_decls[local].local_info;
                if let LocalInfo::StaticRef { def_id, .. } = *local_info {
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
                            let name = self.upvars[var_index].name.to_string();
                            if self.upvars[var_index].by_ref {
                                buf.push_str(&name);
                            } else {
                                buf.push_str(&format!("*{}", &name));
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
                                buf.push_str(&"*");
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

                        let upvar_field_projection = self.is_upvar_field_projection(place);
                        if let Some(field) = upvar_field_projection {
                            let var_index = field.index();
                            let name = self.upvars[var_index].name.to_string();
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
                            buf.push_str(&format!(".{}", field_name));
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
                        buf.push_str("[");
                        if self.append_local_to_string(*index, buf).is_err() {
                            buf.push_str("_");
                        }
                        buf.push_str("]");
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
                        buf.push_str(&"[..]");
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
    fn describe_field(&self, place: PlaceRef<'cx, 'tcx>, field: Field) -> String {
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
                    let base_ty =
                        Place::ty_from(place.local, place.projection, *self.body, self.infcx.tcx)
                            .ty;
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
            match ty.kind {
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
                    // `tcx.upvars(def_id)` returns an `Option`, which is `None` in case
                    // the closure comes from another crate. But in that case we wouldn't
                    // be borrowck'ing it, so we can just unwrap:
                    let (&var_id, _) =
                        self.infcx.tcx.upvars(def_id).unwrap().get_index(field.index()).unwrap();

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
    ) {
        let message = format!(
            "move occurs because {} has type `{}`, which does not implement the `Copy` trait",
            place_desc, ty,
        );
        if let Some(span) = span {
            err.span_label(span, message);
        } else {
            err.note(&message);
        }
    }

    pub(super) fn borrowed_content_source(
        &self,
        deref_base: PlaceRef<'cx, 'tcx>,
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
                            BorrowedContentSource::from_call(func.ty(*self.body, tcx), tcx)
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
        let base_ty = Place::ty_from(deref_base.local, deref_base.projection, *self.body, tcx).ty;
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
        match ty.kind {
            ty::Ref(ty::RegionKind::ReLateBound(_, br), _, _)
            | ty::Ref(
                ty::RegionKind::RePlaceholder(ty::PlaceholderRegion { name: br, .. }),
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

        let region = match ty.kind {
            ty::Ref(region, _, _) => {
                match region {
                    ty::RegionKind::ReLateBound(_, br)
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

// The span(s) associated to a use of a place.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum UseSpans {
    // The access is caused by capturing a variable for a closure.
    ClosureUse {
        // This is true if the captured variable was from a generator.
        generator_kind: Option<GeneratorKind>,
        // The span of the args of the closure, including the `move` keyword if
        // it's present.
        args_span: Span,
        // The span of the first use of the captured variable inside the closure.
        var_span: Span,
    },
    // This access has a single span associated to it: common case.
    OtherUse(Span),
}

impl UseSpans {
    pub(super) fn args_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { args_span: span, .. } | UseSpans::OtherUse(span) => span,
        }
    }

    pub(super) fn var_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { var_span: span, .. } | UseSpans::OtherUse(span) => span,
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
    pub(super) fn var_span_label(
        self,
        err: &mut DiagnosticBuilder<'_>,
        message: impl Into<String>,
    ) {
        if let UseSpans::ClosureUse { var_span, .. } = self {
            err.span_label(var_span, message);
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
            _ => "".to_string(),
        }
    }

    pub(super) fn or_else<F>(self, if_other: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            closure @ UseSpans::ClosureUse { .. } => closure,
            UseSpans::OtherUse(_) => if_other(),
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
    pub(super) fn describe_for_unnamed_place(&self) -> String {
        match *self {
            BorrowedContentSource::DerefRawPointer => format!("a raw pointer"),
            BorrowedContentSource::DerefSharedRef => format!("a shared reference"),
            BorrowedContentSource::DerefMutableRef => format!("a mutable reference"),
            BorrowedContentSource::OverloadedDeref(ty) => {
                if ty.is_rc() {
                    format!("an `Rc`")
                } else if ty.is_arc() {
                    format!("an `Arc`")
                } else {
                    format!("dereference of `{}`", ty)
                }
            }
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

    pub(super) fn describe_for_immutable_place(&self) -> String {
        match *self {
            BorrowedContentSource::DerefRawPointer => format!("a `*const` pointer"),
            BorrowedContentSource::DerefSharedRef => format!("a `&` reference"),
            BorrowedContentSource::DerefMutableRef => {
                bug!("describe_for_immutable_place: DerefMutableRef isn't immutable")
            }
            BorrowedContentSource::OverloadedDeref(ty) => {
                if ty.is_rc() {
                    format!("an `Rc`")
                } else if ty.is_arc() {
                    format!("an `Arc`")
                } else {
                    format!("a dereference of `{}`", ty)
                }
            }
            BorrowedContentSource::OverloadedIndex(ty) => format!("an index of `{}`", ty),
        }
    }

    fn from_call(func: Ty<'tcx>, tcx: TyCtxt<'tcx>) -> Option<Self> {
        match func.kind {
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
        moved_place: PlaceRef<'cx, 'tcx>, // Could also be an upvar.
        location: Location,
    ) -> UseSpans {
        use self::UseSpans::*;

        let stmt = match self.body[location.block].statements.get(location.statement_index) {
            Some(stmt) => stmt,
            None => return OtherUse(self.body.source_info(location).span),
        };

        debug!("move_spans: moved_place={:?} location={:?} stmt={:?}", moved_place, location, stmt);
        if let StatementKind::Assign(box (_, Rvalue::Aggregate(ref kind, ref places))) = stmt.kind {
            let def_id = match kind {
                box AggregateKind::Closure(def_id, _)
                | box AggregateKind::Generator(def_id, _, _) => def_id,
                _ => return OtherUse(stmt.source_info.span),
            };

            debug!("move_spans: def_id={:?} places={:?}", def_id, places);
            if let Some((args_span, generator_kind, var_span)) =
                self.closure_span(*def_id, moved_place, places)
            {
                return ClosureUse { generator_kind, args_span, var_span };
            }
        }

        OtherUse(stmt.source_info.span)
    }

    /// Finds the span of arguments of a closure (within `maybe_closure_span`)
    /// and its usage of the local assigned at `location`.
    /// This is done by searching in statements succeeding `location`
    /// and originating from `maybe_closure_span`.
    pub(super) fn borrow_spans(&self, use_span: Span, location: Location) -> UseSpans {
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
                if let Some((args_span, generator_kind, var_span)) =
                    self.closure_span(*def_id, Place::from(target).as_ref(), places)
                {
                    return ClosureUse { generator_kind, args_span, var_span };
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

    /// Finds the span of a captured variable within a closure or generator.
    fn closure_span(
        &self,
        def_id: DefId,
        target_place: PlaceRef<'cx, 'tcx>,
        places: &Vec<Operand<'tcx>>,
    ) -> Option<(Span, Option<GeneratorKind>, Span)> {
        debug!(
            "closure_span: def_id={:?} target_place={:?} places={:?}",
            def_id, target_place, places
        );
        let hir_id = self.infcx.tcx.hir().as_local_hir_id(def_id)?;
        let expr = &self.infcx.tcx.hir().expect_expr(hir_id).kind;
        debug!("closure_span: hir_id={:?} expr={:?}", hir_id, expr);
        if let hir::ExprKind::Closure(.., body_id, args_span, _) = expr {
            for (upvar, place) in self.infcx.tcx.upvars(def_id)?.values().zip(places) {
                match place {
                    Operand::Copy(place) | Operand::Move(place)
                        if target_place == place.as_ref() =>
                    {
                        debug!("closure_span: found captured local {:?}", place);
                        let body = self.infcx.tcx.hir().body(*body_id);
                        let generator_kind = body.generator_kind();
                        return Some((*args_span, generator_kind, upvar.span));
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Helper to retrieve span(s) of given borrow from the current MIR
    /// representation
    pub(super) fn retrieve_borrow_spans(&self, borrow: &BorrowData<'_>) -> UseSpans {
        let span = self.body.source_info(borrow.reserve_location).span;
        self.borrow_spans(span, borrow.reserve_location)
    }
}
