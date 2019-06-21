use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::sync::Lrc;

use rustc::ty::query::Providers;
use rustc::ty::{self, TyCtxt};
use rustc::ty::cast::CastTy;
use rustc::hir;
use rustc::hir::Node;
use rustc::hir::def_id::DefId;
use rustc::lint::builtin::{SAFE_EXTERN_STATICS, SAFE_PACKED_BORROWS, UNUSED_UNSAFE};
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor, MutatingUseContext};

use syntax::symbol::{InternedString, sym};

use std::ops::Bound;

use crate::util;

pub struct UnsafetyChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    const_context: bool,
    min_const_fn: bool,
    source_scope_local_data: &'a IndexVec<SourceScope, SourceScopeLocalData>,
    violations: Vec<UnsafetyViolation>,
    source_info: SourceInfo,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    /// Mark an `unsafe` block as used, so we don't lint it.
    used_unsafe: FxHashSet<hir::HirId>,
    inherited_blocks: Vec<(hir::HirId, bool)>,
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn new(
        const_context: bool,
        min_const_fn: bool,
        body: &'a Body<'tcx>,
        source_scope_local_data: &'a IndexVec<SourceScope, SourceScopeLocalData>,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        // sanity check
        if min_const_fn {
            assert!(const_context);
        }
        Self {
            body,
            const_context,
            min_const_fn,
            source_scope_local_data,
            violations: vec![],
            source_info: SourceInfo {
                span: body.span,
                scope: OUTERMOST_SOURCE_SCOPE
            },
            tcx,
            param_env,
            used_unsafe: Default::default(),
            inherited_blocks: vec![],
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnsafetyChecker<'a, 'tcx> {
    fn visit_terminator(&mut self,
                        terminator: &Terminator<'tcx>,
                        location: Location)
    {
        self.source_info = terminator.source_info;
        match terminator.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::SwitchInt { .. } |
            TerminatorKind::Drop { .. } |
            TerminatorKind::Yield { .. } |
            TerminatorKind::Assert { .. } |
            TerminatorKind::DropAndReplace { .. } |
            TerminatorKind::GeneratorDrop |
            TerminatorKind::Resume |
            TerminatorKind::Abort |
            TerminatorKind::Return |
            TerminatorKind::Unreachable |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } => {
                // safe (at least as emitted during MIR construction)
            }

            TerminatorKind::Call { ref func, .. } => {
                let func_ty = func.ty(self.body, self.tcx);
                let sig = func_ty.fn_sig(self.tcx);
                if let hir::Unsafety::Unsafe = sig.unsafety() {
                    self.require_unsafe("call to unsafe function",
                        "consult the function's documentation for information on how to avoid \
                         undefined behavior", UnsafetyViolationKind::GeneralAndConstFn)
                }
            }
        }
        self.super_terminator(terminator, location);
    }

    fn visit_statement(&mut self,
                       statement: &Statement<'tcx>,
                       location: Location)
    {
        self.source_info = statement.source_info;
        match statement.kind {
            StatementKind::Assign(..) |
            StatementKind::FakeRead(..) |
            StatementKind::SetDiscriminant { .. } |
            StatementKind::StorageLive(..) |
            StatementKind::StorageDead(..) |
            StatementKind::Retag { .. } |
            StatementKind::AscribeUserType(..) |
            StatementKind::Nop => {
                // safe (at least as emitted during MIR construction)
            }

            StatementKind::InlineAsm { .. } => {
                self.require_unsafe("use of inline assembly",
                    "inline assembly is entirely unchecked and can cause undefined behavior",
                    UnsafetyViolationKind::General)
            },
        }
        self.super_statement(statement, location);
    }

    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location)
    {
        match rvalue {
            Rvalue::Aggregate(box ref aggregate, _) => {
                match aggregate {
                    &AggregateKind::Array(..) |
                    &AggregateKind::Tuple => {}
                    &AggregateKind::Adt(ref def, ..) => {
                        match self.tcx.layout_scalar_valid_range(def.did) {
                            (Bound::Unbounded, Bound::Unbounded) => {},
                            _ => self.require_unsafe(
                                "initializing type with `rustc_layout_scalar_valid_range` attr",
                                "initializing a layout restricted type's field with a value \
                                outside the valid range is undefined behavior",
                                UnsafetyViolationKind::GeneralAndConstFn,
                            ),
                        }
                    }
                    &AggregateKind::Closure(def_id, _) |
                    &AggregateKind::Generator(def_id, _, _) => {
                        let UnsafetyCheckResult {
                            violations, unsafe_blocks
                        } = self.tcx.unsafety_check_result(def_id);
                        self.register_violations(&violations, &unsafe_blocks);
                    }
                }
            },
            // casting pointers to ints is unsafe in const fn because the const evaluator cannot
            // possibly know what the result of various operations like `address / 2` would be
            // pointers during const evaluation have no integral address, only an abstract one
            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty)
            if self.const_context && self.tcx.features().const_raw_ptr_to_usize_cast => {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_), CastTy::Int(_)) |
                    (CastTy::FnPtr, CastTy::Int(_)) => {
                        self.register_violations(&[UnsafetyViolation {
                            source_info: self.source_info,
                            description: InternedString::intern("cast of pointer to int"),
                            details: InternedString::intern(
                                "casting pointers to integers in constants"),
                            kind: UnsafetyViolationKind::General,
                        }], &[]);
                    },
                    _ => {},
                }
            }
            // raw pointer and fn pointer operations are unsafe as it is not clear whether one
            // pointer would be "less" or "equal" to another, because we cannot know where llvm
            // or the linker will place various statics in memory. Without this information the
            // result of a comparison of addresses would differ between runtime and compile-time.
            Rvalue::BinaryOp(_, ref lhs, _)
            if self.const_context && self.tcx.features().const_compare_raw_pointers => {
                if let ty::RawPtr(_) | ty::FnPtr(..) = lhs.ty(self.body, self.tcx).sty {
                    self.register_violations(&[UnsafetyViolation {
                        source_info: self.source_info,
                        description: InternedString::intern("pointer operation"),
                        details: InternedString::intern("operations on pointers in constants"),
                        kind: UnsafetyViolationKind::General,
                    }], &[]);
                }
            }
            _ => {},
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(&mut self,
                    place: &Place<'tcx>,
                    context: PlaceContext,
                    _location: Location) {
        place.iterate(|place_base, place_projections| {
            match place_base {
                PlaceBase::Local(..) => {
                    // Locals are safe.
                }
                PlaceBase::Static(box Static { kind: StaticKind::Promoted(_), .. }) => {
                    bug!("unsafety checking should happen before promotion")
                }
                PlaceBase::Static(box Static { kind: StaticKind::Static(def_id), .. }) => {
                    if self.tcx.is_mutable_static(*def_id) {
                        self.require_unsafe("use of mutable static",
                            "mutable statics can be mutated by multiple threads: aliasing \
                             violations or data races will cause undefined behavior",
                             UnsafetyViolationKind::General);
                    } else if self.tcx.is_foreign_item(*def_id) {
                        let source_info = self.source_info;
                        let lint_root =
                            self.source_scope_local_data[source_info.scope].lint_root;
                        self.register_violations(&[UnsafetyViolation {
                            source_info,
                            description: InternedString::intern("use of extern static"),
                            details: InternedString::intern(
                                "extern statics are not controlled by the Rust type system: \
                                invalid data, aliasing violations or data races will cause \
                                undefined behavior"),
                            kind: UnsafetyViolationKind::ExternStatic(lint_root)
                        }], &[]);
                    }
                }
            }

            for proj in place_projections {
                if context.is_borrow() {
                    if util::is_disaligned(self.tcx, self.body, self.param_env, place) {
                        let source_info = self.source_info;
                        let lint_root =
                            self.source_scope_local_data[source_info.scope].lint_root;
                        self.register_violations(&[UnsafetyViolation {
                            source_info,
                            description: InternedString::intern("borrow of packed field"),
                            details: InternedString::intern(
                                "fields of packed structs might be misaligned: dereferencing a \
                                misaligned pointer or even just creating a misaligned reference \
                                is undefined behavior"),
                            kind: UnsafetyViolationKind::BorrowPacked(lint_root)
                        }], &[]);
                    }
                }
                let is_borrow_of_interior_mut = context.is_borrow() && !proj.base
                    .ty(self.body, self.tcx)
                    .ty
                    .is_freeze(self.tcx, self.param_env, self.source_info.span);
                // prevent
                // * `&mut x.field`
                // * `x.field = y;`
                // * `&x.field` if `field`'s type has interior mutability
                // because either of these would allow modifying the layout constrained field and
                // insert values that violate the layout constraints.
                if context.is_mutating_use() || is_borrow_of_interior_mut {
                    self.check_mut_borrowing_layout_constrained_field(
                        place, context.is_mutating_use(),
                    );
                }
                let old_source_info = self.source_info;
                if let Place::Base(PlaceBase::Local(local)) = proj.base {
                    if self.body.local_decls[local].internal {
                        // Internal locals are used in the `move_val_init` desugaring.
                        // We want to check unsafety against the source info of the
                        // desugaring, rather than the source info of the RHS.
                        self.source_info = self.body.local_decls[local].source_info;
                    }
                }
                let base_ty = proj.base.ty(self.body, self.tcx).ty;
                match base_ty.sty {
                    ty::RawPtr(..) => {
                        self.require_unsafe("dereference of raw pointer",
                            "raw pointers may be NULL, dangling or unaligned; they can violate \
                             aliasing rules and cause data races: all of these are undefined \
                             behavior", UnsafetyViolationKind::General)
                    }
                    ty::Adt(adt, _) => {
                        if adt.is_union() {
                            if context == PlaceContext::MutatingUse(MutatingUseContext::Store) ||
                                context == PlaceContext::MutatingUse(MutatingUseContext::Drop) ||
                                context == PlaceContext::MutatingUse(
                                    MutatingUseContext::AsmOutput
                                )
                            {
                                let elem_ty = match proj.elem {
                                    ProjectionElem::Field(_, ty) => ty,
                                    _ => span_bug!(
                                        self.source_info.span,
                                        "non-field projection {:?} from union?",
                                        place)
                                };
                                if !elem_ty.is_copy_modulo_regions(
                                    self.tcx,
                                    self.param_env,
                                    self.source_info.span,
                                ) {
                                    self.require_unsafe(
                                        "assignment to non-`Copy` union field",
                                        "the previous content of the field will be dropped, which \
                                         causes undefined behavior if the field was not properly \
                                         initialized", UnsafetyViolationKind::General)
                                } else {
                                    // write to non-move union, safe
                                }
                            } else {
                                self.require_unsafe("access to union field",
                                    "the field may not be properly initialized: using \
                                     uninitialized data will cause undefined behavior",
                                     UnsafetyViolationKind::General)
                            }
                        }
                    }
                    _ => {}
                }
                self.source_info = old_source_info;
            }
        });
    }
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn require_unsafe(
        &mut self,
        description: &'static str,
        details: &'static str,
        kind: UnsafetyViolationKind,
    ) {
        let source_info = self.source_info;
        self.register_violations(&[UnsafetyViolation {
            source_info,
            description: InternedString::intern(description),
            details: InternedString::intern(details),
            kind,
        }], &[]);
    }

    fn register_violations(&mut self,
                           violations: &[UnsafetyViolation],
                           unsafe_blocks: &[(hir::HirId, bool)]) {
        let safety = self.source_scope_local_data[self.source_info.scope].safety;
        let within_unsafe = match safety {
            // `unsafe` blocks are required in safe code
            Safety::Safe => {
                for violation in violations {
                    let mut violation = violation.clone();
                    match violation.kind {
                        UnsafetyViolationKind::GeneralAndConstFn |
                        UnsafetyViolationKind::General => {},
                        UnsafetyViolationKind::BorrowPacked(_) |
                        UnsafetyViolationKind::ExternStatic(_) => if self.min_const_fn {
                            // const fns don't need to be backwards compatible and can
                            // emit these violations as a hard error instead of a backwards
                            // compat lint
                            violation.kind = UnsafetyViolationKind::General;
                        },
                    }
                    if !self.violations.contains(&violation) {
                        self.violations.push(violation)
                    }
                }
                false
            }
            // `unsafe` function bodies allow unsafe without additional unsafe blocks
            Safety::BuiltinUnsafe | Safety::FnUnsafe => true,
            Safety::ExplicitUnsafe(hir_id) => {
                // mark unsafe block as used if there are any unsafe operations inside
                if !violations.is_empty() {
                    self.used_unsafe.insert(hir_id);
                }
                // only some unsafety is allowed in const fn
                if self.min_const_fn {
                    for violation in violations {
                        match violation.kind {
                            // these unsafe things are stable in const fn
                            UnsafetyViolationKind::GeneralAndConstFn => {},
                            // these things are forbidden in const fns
                            UnsafetyViolationKind::General |
                            UnsafetyViolationKind::BorrowPacked(_) |
                            UnsafetyViolationKind::ExternStatic(_) => {
                                let mut violation = violation.clone();
                                // const fns don't need to be backwards compatible and can
                                // emit these violations as a hard error instead of a backwards
                                // compat lint
                                violation.kind = UnsafetyViolationKind::General;
                                if !self.violations.contains(&violation) {
                                    self.violations.push(violation)
                                }
                            },
                        }
                    }
                }
                true
            }
        };
        self.inherited_blocks.extend(unsafe_blocks.iter().map(|&(hir_id, is_used)| {
            (hir_id, is_used && !within_unsafe)
        }));
    }
    fn check_mut_borrowing_layout_constrained_field(
        &mut self,
        mut place: &Place<'tcx>,
        is_mut_use: bool,
    ) {
        while let &Place::Projection(box Projection {
            ref base, ref elem
        }) = place {
            match *elem {
                ProjectionElem::Field(..) => {
                    let ty = base.ty(&self.body.local_decls, self.tcx).ty;
                    match ty.sty {
                        ty::Adt(def, _) => match self.tcx.layout_scalar_valid_range(def.did) {
                            (Bound::Unbounded, Bound::Unbounded) => {},
                            _ => {
                                let (description, details) = if is_mut_use {
                                    (
                                        "mutation of layout constrained field",
                                        "mutating layout constrained fields cannot statically be \
                                        checked for valid values",
                                    )
                                } else {
                                    (
                                        "borrow of layout constrained field with interior \
                                        mutability",
                                        "references to fields of layout constrained fields \
                                        lose the constraints. Coupled with interior mutability, \
                                        the field can be changed to invalid values",
                                    )
                                };
                                let source_info = self.source_info;
                                self.register_violations(&[UnsafetyViolation {
                                    source_info,
                                    description: InternedString::intern(description),
                                    details: InternedString::intern(details),
                                    kind: UnsafetyViolationKind::GeneralAndConstFn,
                                }], &[]);
                            }
                        },
                        _ => {}
                    }
                }
                _ => {}
            }
            place = base;
        }
    }
}

pub(crate) fn provide(providers: &mut Providers<'_>) {
    *providers = Providers {
        unsafety_check_result,
        unsafe_derive_on_repr_packed,
        ..*providers
    };
}

struct UnusedUnsafeVisitor<'a> {
    used_unsafe: &'a FxHashSet<hir::HirId>,
    unsafe_blocks: &'a mut Vec<(hir::HirId, bool)>,
}

impl<'a, 'tcx> hir::intravisit::Visitor<'tcx> for UnusedUnsafeVisitor<'a> {
    fn nested_visit_map<'this>(&'this mut self) ->
        hir::intravisit::NestedVisitorMap<'this, 'tcx>
    {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_block(&mut self, block: &'tcx hir::Block) {
        hir::intravisit::walk_block(self, block);

        if let hir::UnsafeBlock(hir::UserProvided) = block.rules {
            self.unsafe_blocks.push((block.hir_id, self.used_unsafe.contains(&block.hir_id)));
        }
    }
}

fn check_unused_unsafe(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    used_unsafe: &FxHashSet<hir::HirId>,
    unsafe_blocks: &mut Vec<(hir::HirId, bool)>,
) {
    let body_id =
        tcx.hir().as_local_hir_id(def_id).and_then(|hir_id| {
            tcx.hir().maybe_body_owned_by(hir_id)
        });

    let body_id = match body_id {
        Some(body) => body,
        None => {
            debug!("check_unused_unsafe({:?}) - no body found", def_id);
            return
        }
    };
    let body = tcx.hir().body(body_id);
    debug!("check_unused_unsafe({:?}, body={:?}, used_unsafe={:?})",
           def_id, body, used_unsafe);

    let mut visitor =  UnusedUnsafeVisitor { used_unsafe, unsafe_blocks };
    hir::intravisit::Visitor::visit_body(&mut visitor, body);
}

fn unsafety_check_result(tcx: TyCtxt<'_>, def_id: DefId) -> UnsafetyCheckResult {
    debug!("unsafety_violations({:?})", def_id);

    // N.B., this borrow is valid because all the consumers of
    // `mir_built` force this.
    let body = &tcx.mir_built(def_id).borrow();

    let source_scope_local_data = match body.source_scope_local_data {
        ClearCrossCrate::Set(ref data) => data,
        ClearCrossCrate::Clear => {
            debug!("unsafety_violations: {:?} - remote, skipping", def_id);
            return UnsafetyCheckResult {
                violations: Lrc::new([]),
                unsafe_blocks: Lrc::new([])
            }
        }
    };

    let param_env = tcx.param_env(def_id);

    let id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let (const_context, min_const_fn) = match tcx.hir().body_owner_kind(id) {
        hir::BodyOwnerKind::Closure => (false, false),
        hir::BodyOwnerKind::Fn => (tcx.is_const_fn(def_id), tcx.is_min_const_fn(def_id)),
        hir::BodyOwnerKind::Const |
        hir::BodyOwnerKind::Static(_) => (true, false),
    };
    let mut checker = UnsafetyChecker::new(
        const_context, min_const_fn,
        body, source_scope_local_data, tcx, param_env);
    checker.visit_body(body);

    check_unused_unsafe(tcx, def_id, &checker.used_unsafe, &mut checker.inherited_blocks);
    UnsafetyCheckResult {
        violations: checker.violations.into(),
        unsafe_blocks: checker.inherited_blocks.into()
    }
}

fn unsafe_derive_on_repr_packed(tcx: TyCtxt<'_>, def_id: DefId) {
    let lint_hir_id = tcx.hir().as_local_hir_id(def_id).unwrap_or_else(||
        bug!("checking unsafety for non-local def id {:?}", def_id));

    // FIXME: when we make this a hard error, this should have its
    // own error code.
    let message = if tcx.generics_of(def_id).own_requires_monomorphization() {
        "#[derive] can't be used on a #[repr(packed)] struct with \
         type or const parameters (error E0133)".to_string()
    } else {
        "#[derive] can't be used on a #[repr(packed)] struct that \
         does not derive Copy (error E0133)".to_string()
    };
    tcx.lint_hir(SAFE_PACKED_BORROWS,
                 lint_hir_id,
                 tcx.def_span(def_id),
                 &message);
}

/// Returns the `HirId` for an enclosing scope that is also `unsafe`.
fn is_enclosed(
    tcx: TyCtxt<'_>,
    used_unsafe: &FxHashSet<hir::HirId>,
    id: hir::HirId,
) -> Option<(String, hir::HirId)> {
    let parent_id = tcx.hir().get_parent_node(id);
    if parent_id != id {
        if used_unsafe.contains(&parent_id) {
            Some(("block".to_string(), parent_id))
        } else if let Some(Node::Item(&hir::Item {
            node: hir::ItemKind::Fn(_, header, _, _),
            ..
        })) = tcx.hir().find(parent_id) {
            match header.unsafety {
                hir::Unsafety::Unsafe => Some(("fn".to_string(), parent_id)),
                hir::Unsafety::Normal => None,
            }
        } else {
            is_enclosed(tcx, used_unsafe, parent_id)
        }
    } else {
        None
    }
}

fn report_unused_unsafe(tcx: TyCtxt<'_>, used_unsafe: &FxHashSet<hir::HirId>, id: hir::HirId) {
    let span = tcx.sess.source_map().def_span(tcx.hir().span(id));
    let msg = "unnecessary `unsafe` block";
    let mut db = tcx.struct_span_lint_hir(UNUSED_UNSAFE, id, span, msg);
    db.span_label(span, msg);
    if let Some((kind, id)) = is_enclosed(tcx, used_unsafe, id) {
        db.span_label(tcx.sess.source_map().def_span(tcx.hir().span(id)),
                      format!("because it's nested under this `unsafe` {}", kind));
    }
    db.emit();
}

fn builtin_derive_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    debug!("builtin_derive_def_id({:?})", def_id);
    if let Some(impl_def_id) = tcx.impl_of_method(def_id) {
        if tcx.has_attr(impl_def_id, sym::automatically_derived) {
            debug!("builtin_derive_def_id({:?}) - is {:?}", def_id, impl_def_id);
            Some(impl_def_id)
        } else {
            debug!("builtin_derive_def_id({:?}) - not automatically derived", def_id);
            None
        }
    } else {
        debug!("builtin_derive_def_id({:?}) - not a method", def_id);
        None
    }
}

pub fn check_unsafety(tcx: TyCtxt<'_>, def_id: DefId) {
    debug!("check_unsafety({:?})", def_id);

    // closures are handled by their parent fn.
    if tcx.is_closure(def_id) {
        return;
    }

    let UnsafetyCheckResult {
        violations,
        unsafe_blocks
    } = tcx.unsafety_check_result(def_id);

    for &UnsafetyViolation {
        source_info, description, details, kind
    } in violations.iter() {
        // Report an error.
        match kind {
            UnsafetyViolationKind::GeneralAndConstFn |
            UnsafetyViolationKind::General => {
                struct_span_err!(
                    tcx.sess, source_info.span, E0133,
                    "{} is unsafe and requires unsafe function or block", description)
                    .span_label(source_info.span, &description.as_str()[..])
                    .note(&details.as_str()[..])
                    .emit();
            }
            UnsafetyViolationKind::ExternStatic(lint_hir_id) => {
                tcx.lint_node_note(SAFE_EXTERN_STATICS,
                              lint_hir_id,
                              source_info.span,
                              &format!("{} is unsafe and requires unsafe function or block \
                                        (error E0133)", &description.as_str()[..]),
                              &details.as_str()[..]);
            }
            UnsafetyViolationKind::BorrowPacked(lint_hir_id) => {
                if let Some(impl_def_id) = builtin_derive_def_id(tcx, def_id) {
                    tcx.unsafe_derive_on_repr_packed(impl_def_id);
                } else {
                    tcx.lint_node_note(SAFE_PACKED_BORROWS,
                                  lint_hir_id,
                                  source_info.span,
                                  &format!("{} is unsafe and requires unsafe function or block \
                                            (error E0133)", &description.as_str()[..]),
                                  &details.as_str()[..]);
                }
            }
        }
    }

    let mut unsafe_blocks: Vec<_> = unsafe_blocks.into_iter().collect();
    unsafe_blocks.sort_by_cached_key(|(hir_id, _)| tcx.hir().hir_to_node_id(*hir_id));
    let used_unsafe: FxHashSet<_> = unsafe_blocks.iter()
        .flat_map(|&&(id, used)| if used { Some(id) } else { None })
        .collect();
    for &(block_id, is_used) in unsafe_blocks {
        if !is_used {
            report_unused_unsafe(tcx, &used_unsafe, block_id);
        }
    }
}
