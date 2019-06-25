use rustc::hir;
use rustc::hir::def::Namespace;
use rustc::hir::def_id::DefId;
use rustc::mir::{
    AggregateKind, Constant, Field, Local, LocalKind, Location, Operand,
    Place, PlaceBase, ProjectionElem, Rvalue, Statement, StatementKind, Static,
    StaticKind, TerminatorKind,
};
use rustc::ty::{self, DefIdTree, Ty};
use rustc::ty::layout::VariantIdx;
use rustc::ty::print::Print;
use rustc_errors::DiagnosticBuilder;
use syntax_pos::Span;
use syntax::symbol::sym;

use super::borrow_set::BorrowData;
use super::{MirBorrowckCtxt};

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
        place: &Place<'tcx>,
        diag: &mut DiagnosticBuilder<'_>,
    ) {
        debug!("add_moved_or_invoked_closure_note: location={:?} place={:?}", location, place);
        let mut target = place.local_or_deref_local();
        for stmt in &self.body[location.block].statements[location.statement_index..] {
            debug!("add_moved_or_invoked_closure_note: stmt={:?} target={:?}", stmt, target);
            if let StatementKind::Assign(into, box Rvalue::Use(from)) = &stmt.kind {
                debug!("add_fnonce_closure_note: into={:?} from={:?}", into, from);
                match from {
                    Operand::Copy(ref place) |
                    Operand::Move(ref place) if target == place.local_or_deref_local() =>
                        target = into.local_or_deref_local(),
                    _ => {},
                }
            }
        }

        // Check if we are attempting to call a closure after it has been invoked.
        let terminator = self.body[location.block].terminator();
        debug!("add_moved_or_invoked_closure_note: terminator={:?}", terminator);
        if let TerminatorKind::Call {
            func: Operand::Constant(box Constant {
                literal: ty::Const {
                    ty: &ty::TyS { sty: ty::FnDef(id, _), ..  },
                    ..
                },
                ..
            }),
            args,
            ..
        } = &terminator.kind {
            debug!("add_moved_or_invoked_closure_note: id={:?}", id);
            if self.infcx.tcx.parent(id) == self.infcx.tcx.lang_items().fn_once_trait() {
                let closure = match args.first() {
                    Some(Operand::Copy(ref place)) |
                    Some(Operand::Move(ref place)) if target == place.local_or_deref_local() =>
                        place.local_or_deref_local().unwrap(),
                    _ => return,
                };

                debug!("add_moved_or_invoked_closure_note: closure={:?}", closure);
                if let ty::Closure(did, _) = self.body.local_decls[closure].ty.sty {
                    let hir_id = self.infcx.tcx.hir().as_local_hir_id(did).unwrap();

                    if let Some((span, name)) = self.infcx.tcx.typeck_tables_of(did)
                        .closure_kind_origins()
                        .get(hir_id)
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
            if let ty::Closure(did, _) = self.body.local_decls[target].ty.sty {
                let hir_id = self.infcx.tcx.hir().as_local_hir_id(did).unwrap();

                if let Some((span, name)) = self.infcx.tcx.typeck_tables_of(did)
                    .closure_kind_origins()
                    .get(hir_id)
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
    pub(super) fn describe_place(&self, place: &Place<'tcx>) -> Option<String> {
        self.describe_place_with_options(place, IncludingDowncast(false))
    }

    /// End-user visible description of `place` if one can be found. If the
    /// place is a temporary for instance, None will be returned.
    /// `IncludingDowncast` parameter makes the function return `Err` if `ProjectionElem` is
    /// `Downcast` and `IncludingDowncast` is true
    pub(super) fn describe_place_with_options(
        &self,
        place: &Place<'tcx>,
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
        place: &Place<'tcx>,
        buf: &mut String,
        mut autoderef: bool,
        including_downcast: &IncludingDowncast,
    ) -> Result<(), ()> {
        match *place {
            Place::Base(PlaceBase::Local(local)) => {
                self.append_local_to_string(local, buf)?;
            }
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Promoted(_), .. })) => {
                buf.push_str("promoted");
            }
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Static(def_id), .. })) => {
                buf.push_str(&self.infcx.tcx.item_name(def_id).to_string());
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let upvar_field_projection =
                            self.is_upvar_field_projection(place);
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
                                self.append_place_to_string(
                                    &proj.base,
                                    buf,
                                    autoderef,
                                    &including_downcast,
                                )?;
                            } else if let Place::Base(PlaceBase::Local(local)) = proj.base {
                                if self.body.local_decls[local].is_ref_for_guard() {
                                    self.append_place_to_string(
                                        &proj.base,
                                        buf,
                                        autoderef,
                                        &including_downcast,
                                    )?;
                                } else {
                                    buf.push_str(&"*");
                                    self.append_place_to_string(
                                        &proj.base,
                                        buf,
                                        autoderef,
                                        &including_downcast,
                                    )?;
                                }
                            } else {
                                buf.push_str(&"*");
                                self.append_place_to_string(
                                    &proj.base,
                                    buf,
                                    autoderef,
                                    &including_downcast,
                                )?;
                            }
                        }
                    }
                    ProjectionElem::Downcast(..) => {
                        self.append_place_to_string(
                            &proj.base,
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

                        let upvar_field_projection =
                            self.is_upvar_field_projection(place);
                        if let Some(field) = upvar_field_projection {
                            let var_index = field.index();
                            let name = self.upvars[var_index].name.to_string();
                            buf.push_str(&name);
                        } else {
                            let field_name = self.describe_field(&proj.base, field);
                            self.append_place_to_string(
                                &proj.base,
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
                            &proj.base,
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        buf.push_str("[");
                        if self.append_local_to_string(index, buf).is_err() {
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
                            &proj.base,
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
    fn append_local_to_string(&self, local_index: Local, buf: &mut String) -> Result<(), ()> {
        let local = &self.body.local_decls[local_index];
        match local.name {
            Some(name) if !local.from_compiler_desugaring() => {
                buf.push_str(name.as_str().get());
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// End-user visible description of the `field`nth field of `base`
    fn describe_field(&self, base: &Place<'tcx>, field: Field) -> String {
        match *base {
            Place::Base(PlaceBase::Local(local)) => {
                let local = &self.body.local_decls[local];
                self.describe_field_from_ty(&local.ty, field, None)
            }
            Place::Base(PlaceBase::Static(ref static_)) =>
                self.describe_field_from_ty(&static_.ty, field, None),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Deref => self.describe_field(&proj.base, field),
                ProjectionElem::Downcast(_, variant_index) => {
                    let base_ty = base.ty(self.body, self.infcx.tcx).ty;
                    self.describe_field_from_ty(&base_ty, field, Some(variant_index))
                }
                ProjectionElem::Field(_, field_type) => {
                    self.describe_field_from_ty(&field_type, field, None)
                }
                ProjectionElem::Index(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {
                    self.describe_field(&proj.base, field)
                }
            },
        }
    }

    /// End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(
        &self,
        ty: Ty<'_>,
        field: Field,
        variant_index: Option<VariantIdx>
    ) -> String {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(&ty.boxed_ty(), field, variant_index)
        } else {
            match ty.sty {
                ty::Adt(def, _) => {
                    let variant = if let Some(idx) = variant_index {
                        assert!(def.is_enum());
                        &def.variants[idx]
                    } else {
                        def.non_enum_variant()
                    };
                    variant.fields[field.index()]
                        .ident
                        .to_string()
                },
                ty::Tuple(_) => field.index().to_string(),
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    self.describe_field_from_ty(&ty, field, variant_index)
                }
                ty::Array(ty, _) | ty::Slice(ty) =>
                    self.describe_field_from_ty(&ty, field, variant_index),
                ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                    // `tcx.upvars(def_id)` returns an `Option`, which is `None` in case
                    // the closure comes from another crate. But in that case we wouldn't
                    // be borrowck'ing it, so we can just unwrap:
                    let (&var_id, _) = self.infcx.tcx.upvars(def_id).unwrap()
                        .get_index(field.index()).unwrap();

                    self.infcx.tcx.hir().name(var_id).to_string()
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!(
                        "End-user description not implemented for field access on `{:?}`",
                        ty
                    );
                }
            }
        }
    }

    /// Checks if a place is a thread-local static.
    pub fn is_place_thread_local(&self, place: &Place<'tcx>) -> bool {
        if let Place::Base(
            PlaceBase::Static(box Static{ kind: StaticKind::Static(def_id), .. })
        ) = place {
            let attrs = self.infcx.tcx.get_attrs(*def_id);
            let is_thread_local = attrs.iter().any(|attr| attr.check_name(sym::thread_local));

            debug!(
                "is_place_thread_local: attrs={:?} is_thread_local={:?}",
                attrs, is_thread_local
            );
            is_thread_local
        } else {
            debug!("is_place_thread_local: no");
            false
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
            place_desc,
            ty,
        );
        if let Some(span) = span {
            err.span_label(span, message);
        } else {
            err.note(&message);
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
        match ty.sty {
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

        let region = match ty.sty {
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
        is_generator: bool,
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
            UseSpans::ClosureUse {
                args_span: span, ..
            }
            | UseSpans::OtherUse(span) => span,
        }
    }

    pub(super) fn var_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { var_span: span, .. } | UseSpans::OtherUse(span) => span,
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
            UseSpans::ClosureUse { is_generator, .. } => !is_generator,
            _ => false,
        }
    }

    /// Returns `false` if this place is not used in a generator.
    pub(super) fn for_generator(&self) -> bool {
        match *self {
            UseSpans::ClosureUse { is_generator, .. } => is_generator,
            _ => false,
        }
    }

    /// Describe the span associated with a use of a place.
    pub(super) fn describe(&self) -> String {
        match *self {
            UseSpans::ClosureUse { is_generator, .. } => if is_generator {
                " in generator".to_string()
            } else {
                " in closure".to_string()
            },
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

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    /// Finds the spans associated to a move or copy of move_place at location.
    pub(super) fn move_spans(
        &self,
        moved_place: &Place<'tcx>, // Could also be an upvar.
        location: Location,
    ) -> UseSpans {
        use self::UseSpans::*;

        let stmt = match self.body[location.block].statements.get(location.statement_index) {
            Some(stmt) => stmt,
            None => return OtherUse(self.body.source_info(location).span),
        };

        debug!("move_spans: moved_place={:?} location={:?} stmt={:?}", moved_place, location, stmt);
        if let  StatementKind::Assign(
            _,
            box Rvalue::Aggregate(ref kind, ref places)
        ) = stmt.kind {
            let (def_id, is_generator) = match kind {
                box AggregateKind::Closure(def_id, _) => (def_id, false),
                box AggregateKind::Generator(def_id, _, _) => (def_id, true),
                _ => return OtherUse(stmt.source_info.span),
            };

            debug!(
                "move_spans: def_id={:?} is_generator={:?} places={:?}",
                def_id, is_generator, places
            );
            if let Some((args_span, var_span)) = self.closure_span(*def_id, moved_place, places) {
                return ClosureUse {
                    is_generator,
                    args_span,
                    var_span,
                };
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

        let target = match self.body[location.block]
            .statements
            .get(location.statement_index)
        {
            Some(&Statement {
                kind: StatementKind::Assign(Place::Base(PlaceBase::Local(local)), _),
                ..
            }) => local,
            _ => return OtherUse(use_span),
        };

        if self.body.local_kind(target) != LocalKind::Temp {
            // operands are always temporaries.
            return OtherUse(use_span);
        }

        for stmt in &self.body[location.block].statements[location.statement_index + 1..] {
            if let StatementKind::Assign(
                _, box Rvalue::Aggregate(ref kind, ref places)
            ) = stmt.kind {
                let (def_id, is_generator) = match kind {
                    box AggregateKind::Closure(def_id, _) => (def_id, false),
                    box AggregateKind::Generator(def_id, _, _) => (def_id, true),
                    _ => continue,
                };

                debug!(
                    "borrow_spans: def_id={:?} is_generator={:?} places={:?}",
                    def_id, is_generator, places
                );
                if let Some((args_span, var_span)) = self.closure_span(
                    *def_id, &Place::from(target), places
                ) {
                    return ClosureUse {
                        is_generator,
                        args_span,
                        var_span,
                    };
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
        target_place: &Place<'tcx>,
        places: &Vec<Operand<'tcx>>,
    ) -> Option<(Span, Span)> {
        debug!(
            "closure_span: def_id={:?} target_place={:?} places={:?}",
            def_id, target_place, places
        );
        let hir_id = self.infcx.tcx.hir().as_local_hir_id(def_id)?;
        let expr = &self.infcx.tcx.hir().expect_expr(hir_id).node;
        debug!("closure_span: hir_id={:?} expr={:?}", hir_id, expr);
        if let hir::ExprKind::Closure(
            .., args_span, _
        ) = expr {
            for (upvar, place) in self.infcx.tcx.upvars(def_id)?.values().zip(places) {
                match place {
                    Operand::Copy(place) |
                    Operand::Move(place) if target_place == place => {
                        debug!("closure_span: found captured local {:?}", place);
                        return Some((*args_span, upvar.span));
                    },
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
