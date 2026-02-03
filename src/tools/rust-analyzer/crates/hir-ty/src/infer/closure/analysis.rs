//! Post-inference closure analysis: captures and closure kind.

use std::{cmp, mem};

use base_db::Crate;
use hir_def::{
    DefWithBodyId, FieldId, HasModule, VariantId,
    expr_store::path::Path,
    hir::{
        Array, AsmOperand, BinaryOp, BindingId, CaptureBy, Expr, ExprId, ExprOrPatId, Pat, PatId,
        RecordSpread, Statement, UnaryOp,
    },
    item_tree::FieldsShape,
    resolver::ValueNs,
};
use rustc_ast_ir::Mutability;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::inherent::{GenericArgs as _, IntoKind, Ty as _};
use smallvec::{SmallVec, smallvec};
use stdx::{format_to, never};
use syntax::utils::is_raw_identifier;

use crate::{
    Adjust, Adjustment, BindingMode,
    db::{HirDatabase, InternedClosure, InternedClosureId},
    display::{DisplayTarget, HirDisplay as _},
    infer::InferenceContext,
    mir::{BorrowKind, MirSpan, MutBorrowKind},
    next_solver::{
        DbInterner, ErrorGuaranteed, GenericArgs, ParamEnv, StoredEarlyBinder, StoredTy, Ty,
        TyKind,
        infer::{InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
    traits::FnTrait,
};

// The below functions handle capture and closure kind (Fn, FnMut, ..)

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum HirPlaceProjection {
    Deref,
    Field(FieldId),
    TupleField(u32),
}

impl HirPlaceProjection {
    fn projected_ty<'db>(
        self,
        infcx: &InferCtxt<'db>,
        env: ParamEnv<'db>,
        mut base: Ty<'db>,
        krate: Crate,
    ) -> Ty<'db> {
        let interner = infcx.interner;
        let db = interner.db;
        if base.is_ty_error() {
            return Ty::new_error(interner, ErrorGuaranteed);
        }

        if matches!(base.kind(), TyKind::Alias(..)) {
            let mut ocx = ObligationCtxt::new(infcx);
            match ocx.structurally_normalize_ty(&ObligationCause::dummy(), env, base) {
                Ok(it) => base = it,
                Err(_) => return Ty::new_error(interner, ErrorGuaranteed),
            }
        }
        match self {
            HirPlaceProjection::Deref => match base.kind() {
                TyKind::RawPtr(inner, _) | TyKind::Ref(_, inner, _) => inner,
                TyKind::Adt(adt_def, subst) if adt_def.is_box() => subst.type_at(0),
                _ => {
                    never!(
                        "Overloaded deref on type {} is not a projection",
                        base.display(db, DisplayTarget::from_crate(db, krate))
                    );
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            HirPlaceProjection::Field(f) => match base.kind() {
                TyKind::Adt(_, subst) => {
                    db.field_types(f.parent)[f.local_id].get().instantiate(interner, subst)
                }
                ty => {
                    never!("Only adt has field, found {:?}", ty);
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
            HirPlaceProjection::TupleField(idx) => match base.kind() {
                TyKind::Tuple(subst) => {
                    subst.as_slice().get(idx as usize).copied().unwrap_or_else(|| {
                        never!("Out of bound tuple field");
                        Ty::new_error(interner, ErrorGuaranteed)
                    })
                }
                ty => {
                    never!("Only tuple has tuple field: {:?}", ty);
                    Ty::new_error(interner, ErrorGuaranteed)
                }
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, salsa::Update)]
pub(crate) struct HirPlace {
    pub(crate) local: BindingId,
    pub(crate) projections: Vec<HirPlaceProjection>,
}

impl HirPlace {
    fn ty<'db>(&self, ctx: &mut InferenceContext<'_, 'db>) -> Ty<'db> {
        let krate = ctx.krate();
        let mut ty = ctx.table.resolve_completely(ctx.result.binding_ty(self.local));
        for p in &self.projections {
            ty = p.projected_ty(ctx.infcx(), ctx.table.param_env, ty, krate);
        }
        ty
    }

    fn capture_kind_of_truncated_place(
        &self,
        mut current_capture: CaptureKind,
        len: usize,
    ) -> CaptureKind {
        if let CaptureKind::ByRef(BorrowKind::Mut {
            kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow,
        }) = current_capture
            && self.projections[len..].contains(&HirPlaceProjection::Deref)
        {
            current_capture =
                CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture });
        }
        current_capture
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureKind {
    ByRef(BorrowKind),
    ByValue,
}

#[derive(Debug, Clone, PartialEq, Eq, salsa::Update)]
pub struct CapturedItem {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    ///
    /// Even though we always report only the last span (i.e. the most inclusive span),
    /// we need to keep them all, since when a closure occurs inside a closure, we
    /// copy all captures of the inner closure to the outer closure, and then we may
    /// truncate them, and we want the correct span to be reported.
    span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
    pub(crate) ty: StoredEarlyBinder<StoredTy>,
}

impl CapturedItem {
    pub fn local(&self) -> BindingId {
        self.place.local
    }

    /// Returns whether this place has any field (aka. non-deref) projections.
    pub fn has_field_projections(&self) -> bool {
        self.place.projections.iter().any(|it| !matches!(it, HirPlaceProjection::Deref))
    }

    pub fn ty<'db>(&self, db: &'db dyn HirDatabase, subst: GenericArgs<'db>) -> Ty<'db> {
        let interner = DbInterner::new_no_crate(db);
        self.ty.get().instantiate(interner, subst.as_closure().parent_args())
    }

    pub fn kind(&self) -> CaptureKind {
        self.kind
    }

    pub fn spans(&self) -> SmallVec<[MirSpan; 3]> {
        self.span_stacks.iter().map(|stack| *stack.last().expect("empty span stack")).collect()
    }

    /// Converts the place to a name that can be inserted into source code.
    pub fn place_to_name(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let mut result = body[self.place.local].name.as_str().to_owned();
        for proj in &self.place.projections {
            match proj {
                HirPlaceProjection::Deref => {}
                HirPlaceProjection::Field(f) => {
                    let variant_data = f.parent.fields(db);
                    match variant_data.shape {
                        FieldsShape::Record => {
                            result.push('_');
                            result.push_str(variant_data.fields()[f.local_id].name.as_str())
                        }
                        FieldsShape::Tuple => {
                            let index =
                                variant_data.fields().iter().position(|it| it.0 == f.local_id);
                            if let Some(index) = index {
                                format_to!(result, "_{index}");
                            }
                        }
                        FieldsShape::Unit => {}
                    }
                }
                HirPlaceProjection::TupleField(idx) => {
                    format_to!(result, "_{idx}")
                }
            }
        }
        if is_raw_identifier(&result, owner.module(db).krate(db).data(db).edition) {
            result.insert_str(0, "r#");
        }
        result
    }

    pub fn display_place_source_code(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db);
        let edition = krate.data(db).edition;
        let mut result = body[self.place.local].name.display(db, edition).to_string();
        for proj in &self.place.projections {
            match proj {
                // In source code autoderef kicks in.
                HirPlaceProjection::Deref => {}
                HirPlaceProjection::Field(f) => {
                    let variant_data = f.parent.fields(db);
                    match variant_data.shape {
                        FieldsShape::Record => format_to!(
                            result,
                            ".{}",
                            variant_data.fields()[f.local_id].name.display(db, edition)
                        ),
                        FieldsShape::Tuple => format_to!(
                            result,
                            ".{}",
                            variant_data
                                .fields()
                                .iter()
                                .position(|it| it.0 == f.local_id)
                                .unwrap_or_default()
                        ),
                        FieldsShape::Unit => {}
                    }
                }
                HirPlaceProjection::TupleField(idx) => {
                    format_to!(result, ".{idx}")
                }
            }
        }
        let final_derefs_count = self
            .place
            .projections
            .iter()
            .rev()
            .take_while(|proj| matches!(proj, HirPlaceProjection::Deref))
            .count();
        result.insert_str(0, &"*".repeat(final_derefs_count));
        result
    }

    pub fn display_place(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db);
        let edition = krate.data(db).edition;
        let mut result = body[self.place.local].name.display(db, edition).to_string();
        let mut field_need_paren = false;
        for proj in &self.place.projections {
            match proj {
                HirPlaceProjection::Deref => {
                    result = format!("*{result}");
                    field_need_paren = true;
                }
                HirPlaceProjection::Field(f) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    let variant_data = f.parent.fields(db);
                    let field = match variant_data.shape {
                        FieldsShape::Record => {
                            variant_data.fields()[f.local_id].name.as_str().to_owned()
                        }
                        FieldsShape::Tuple => variant_data
                            .fields()
                            .iter()
                            .position(|it| it.0 == f.local_id)
                            .unwrap_or_default()
                            .to_string(),
                        FieldsShape::Unit => "[missing field]".to_owned(),
                    };
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                HirPlaceProjection::TupleField(idx) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{idx}");
                    field_need_paren = false;
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapturedItemWithoutTy {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    pub(crate) span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
}

impl CapturedItemWithoutTy {
    fn with_ty(self, ctx: &mut InferenceContext<'_, '_>) -> CapturedItem {
        let ty = self.place.ty(ctx);
        let ty = match &self.kind {
            CaptureKind::ByValue => ty,
            CaptureKind::ByRef(bk) => {
                let m = match bk {
                    BorrowKind::Mut { .. } => Mutability::Mut,
                    _ => Mutability::Not,
                };
                Ty::new_ref(ctx.interner(), ctx.types.regions.error, ty, m)
            }
        };
        CapturedItem {
            place: self.place,
            kind: self.kind,
            span_stacks: self.span_stacks,
            ty: StoredEarlyBinder::bind(ty.store()),
        }
    }
}

impl<'db> InferenceContext<'_, 'db> {
    fn place_of_expr(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        let r = self.place_of_expr_without_adjust(tgt_expr)?;
        let adjustments =
            self.result.expr_adjustments.get(&tgt_expr).map(|it| &**it).unwrap_or_default();
        apply_adjusts_to_place(&mut self.current_capture_span_stack, r, adjustments)
    }

    /// Pushes the span into `current_capture_span_stack`, *without clearing it first*.
    fn path_place(&mut self, path: &Path, id: ExprOrPatId) -> Option<HirPlace> {
        if path.type_anchor().is_some() {
            return None;
        }
        let hygiene = self.body.expr_or_pat_path_hygiene(id);
        self.resolver.resolve_path_in_value_ns_fully(self.db, path, hygiene).and_then(|result| {
            match result {
                ValueNs::LocalBinding(binding) => {
                    let mir_span = match id {
                        ExprOrPatId::ExprId(id) => MirSpan::ExprId(id),
                        ExprOrPatId::PatId(id) => MirSpan::PatId(id),
                    };
                    self.current_capture_span_stack.push(mir_span);
                    Some(HirPlace { local: binding, projections: Vec::new() })
                }
                _ => None,
            }
        })
    }

    /// Changes `current_capture_span_stack` to contain the stack of spans for this expr.
    fn place_of_expr_without_adjust(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        self.current_capture_span_stack.clear();
        match &self.body[tgt_expr] {
            Expr::Path(p) => {
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                let result = self.path_place(p, tgt_expr.into());
                self.resolver.reset_to_guard(resolver_guard);
                return result;
            }
            Expr::Field { expr, name: _ } => {
                let mut place = self.place_of_expr(*expr)?;
                let field = self.result.field_resolution(tgt_expr)?;
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                place.projections.push(field.either(HirPlaceProjection::Field, |f| {
                    HirPlaceProjection::TupleField(f.index)
                }));
                return Some(place);
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                let is_builtin_deref = match self.expr_ty(*expr).kind() {
                    TyKind::Ref(..) | TyKind::RawPtr(..) => true,
                    TyKind::Adt(adt_def, _) if adt_def.is_box() => true,
                    _ => false,
                };
                if is_builtin_deref {
                    let mut place = self.place_of_expr(*expr)?;
                    self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                    place.projections.push(HirPlaceProjection::Deref);
                    return Some(place);
                }
            }
            _ => (),
        }
        None
    }

    fn push_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        self.current_captures.push(CapturedItemWithoutTy {
            place,
            kind,
            span_stacks: smallvec![self.current_capture_span_stack.iter().copied().collect()],
        });
    }

    fn truncate_capture_spans(&self, capture: &mut CapturedItemWithoutTy, mut truncate_to: usize) {
        // The first span is the identifier, and it must always remain.
        truncate_to += 1;
        for span_stack in &mut capture.span_stacks {
            let mut remained = truncate_to;
            let mut actual_truncate_to = 0;
            for &span in &*span_stack {
                actual_truncate_to += 1;
                if !span.is_ref_span(self.body) {
                    remained -= 1;
                    if remained == 0 {
                        break;
                    }
                }
            }
            if actual_truncate_to < span_stack.len()
                && span_stack[actual_truncate_to].is_ref_span(self.body)
            {
                // Include the ref operator if there is one, we will fix it later (in `strip_captures_ref_span()`) if it's incorrect.
                actual_truncate_to += 1;
            }
            span_stack.truncate(actual_truncate_to);
        }
    }

    fn ref_expr(&mut self, expr: ExprId, place: Option<HirPlace>) {
        if let Some(place) = place {
            self.add_capture(place, CaptureKind::ByRef(BorrowKind::Shared));
        }
        self.walk_expr(expr);
    }

    fn add_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        if self.is_upvar(&place) {
            self.push_capture(place, kind);
        }
    }

    fn mutate_path_pat(&mut self, path: &Path, id: PatId) {
        if let Some(place) = self.path_place(path, id.into()) {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            );
            self.current_capture_span_stack.pop(); // Remove the pattern span.
        }
    }

    fn mutate_expr(&mut self, expr: ExprId, place: Option<HirPlace>) {
        if let Some(place) = place {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            );
        }
        self.walk_expr(expr);
    }

    fn consume_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.consume_place(place);
        }
        self.walk_expr(expr);
    }

    fn consume_place(&mut self, place: HirPlace) {
        if self.is_upvar(&place) {
            let ty = place.ty(self);
            let kind = if self.is_ty_copy(ty) {
                CaptureKind::ByRef(BorrowKind::Shared)
            } else {
                CaptureKind::ByValue
            };
            self.push_capture(place, kind);
        }
    }

    fn walk_expr_with_adjust(&mut self, tgt_expr: ExprId, adjustment: &[Adjustment]) {
        if let Some((last, rest)) = adjustment.split_last() {
            match &last.kind {
                Adjust::NeverToAny | Adjust::Deref(None) | Adjust::Pointer(_) => {
                    self.walk_expr_with_adjust(tgt_expr, rest)
                }
                Adjust::Deref(Some(m)) => match m.0 {
                    Some(m) => {
                        self.ref_capture_with_adjusts(m, tgt_expr, rest);
                    }
                    None => unreachable!(),
                },
                Adjust::Borrow(b) => {
                    self.ref_capture_with_adjusts(b.mutability(), tgt_expr, rest);
                }
            }
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn ref_capture_with_adjusts(&mut self, m: Mutability, tgt_expr: ExprId, rest: &[Adjustment]) {
        let capture_kind = match m {
            Mutability::Mut => CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            Mutability::Not => CaptureKind::ByRef(BorrowKind::Shared),
        };
        if let Some(place) = self.place_of_expr_without_adjust(tgt_expr)
            && let Some(place) =
                apply_adjusts_to_place(&mut self.current_capture_span_stack, place, rest)
        {
            self.add_capture(place, capture_kind);
        }
        self.walk_expr_with_adjust(tgt_expr, rest);
    }

    fn walk_expr(&mut self, tgt_expr: ExprId) {
        if let Some(it) = self.result.expr_adjustments.get_mut(&tgt_expr) {
            // FIXME: this take is completely unneeded, and just is here to make borrow checker
            // happy. Remove it if you can.
            let x_taken = mem::take(it);
            self.walk_expr_with_adjust(tgt_expr, &x_taken);
            *self.result.expr_adjustments.get_mut(&tgt_expr).unwrap() = x_taken;
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn walk_expr_without_adjust(&mut self, tgt_expr: ExprId) {
        match &self.body[tgt_expr] {
            Expr::OffsetOf(_) => (),
            Expr::InlineAsm(e) => e.operands.iter().for_each(|(_, op)| match op {
                AsmOperand::In { expr, .. }
                | AsmOperand::Out { expr: Some(expr), .. }
                | AsmOperand::InOut { expr, .. } => self.walk_expr_without_adjust(*expr),
                AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    self.walk_expr_without_adjust(*in_expr);
                    if let Some(out_expr) = out_expr {
                        self.walk_expr_without_adjust(*out_expr);
                    }
                }
                AsmOperand::Out { expr: None, .. }
                | AsmOperand::Const(_)
                | AsmOperand::Label(_)
                | AsmOperand::Sym(_) => (),
            }),
            Expr::If { condition, then_branch, else_branch } => {
                self.consume_expr(*condition);
                self.consume_expr(*then_branch);
                if let &Some(expr) = else_branch {
                    self.consume_expr(expr);
                }
            }
            Expr::Async { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Block { statements, tail, .. } => {
                for s in statements.iter() {
                    match s {
                        Statement::Let { pat, type_ref: _, initializer, else_branch } => {
                            if let Some(else_branch) = else_branch {
                                self.consume_expr(*else_branch);
                            }
                            if let Some(initializer) = initializer {
                                if else_branch.is_some() {
                                    self.consume_expr(*initializer);
                                } else {
                                    self.walk_expr(*initializer);
                                }
                                if let Some(place) = self.place_of_expr(*initializer) {
                                    self.consume_with_pat(place, *pat);
                                }
                            }
                        }
                        Statement::Expr { expr, has_semi: _ } => {
                            self.consume_expr(*expr);
                        }
                        Statement::Item(_) => (),
                    }
                }
                if let Some(tail) = tail {
                    self.consume_expr(*tail);
                }
            }
            Expr::Call { callee, args } => {
                self.consume_expr(*callee);
                self.consume_exprs(args.iter().copied());
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.consume_expr(*receiver);
                self.consume_exprs(args.iter().copied());
            }
            Expr::Match { expr, arms } => {
                for arm in arms.iter() {
                    self.consume_expr(arm.expr);
                    if let Some(guard) = arm.guard {
                        self.consume_expr(guard);
                    }
                }
                self.walk_expr(*expr);
                if let Some(discr_place) = self.place_of_expr(*expr)
                    && self.is_upvar(&discr_place)
                {
                    let mut capture_mode = None;
                    for arm in arms.iter() {
                        self.walk_pat(&mut capture_mode, arm.pat);
                    }
                    if let Some(c) = capture_mode {
                        self.push_capture(discr_place, c);
                    }
                }
            }
            Expr::Break { expr, label: _ }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    self.consume_expr(expr);
                }
            }
            &Expr::Become { expr } => {
                self.consume_expr(expr);
            }
            Expr::RecordLit { fields, spread, .. } => {
                if let RecordSpread::Expr(expr) = *spread {
                    self.consume_expr(expr);
                }
                self.consume_exprs(fields.iter().map(|it| it.expr));
            }
            Expr::Field { expr, name: _ } => self.select_from_expr(*expr),
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if self.result.method_resolution(tgt_expr).is_some() {
                    // Overloaded deref.
                    match self.expr_ty_after_adjustments(*expr).kind() {
                        TyKind::Ref(_, _, mutability) => {
                            let place = self.place_of_expr(*expr);
                            match mutability {
                                Mutability::Mut => self.mutate_expr(*expr, place),
                                Mutability::Not => self.ref_expr(*expr, place),
                            }
                        }
                        // FIXME: Is this correct wrt. raw pointer derefs?
                        TyKind::RawPtr(..) => self.select_from_expr(*expr),
                        _ => never!("deref adjustments should include taking a mutable reference"),
                    }
                } else {
                    self.select_from_expr(*expr);
                }
            }
            Expr::Let { pat, expr } => {
                self.walk_expr(*expr);
                if let Some(place) = self.place_of_expr(*expr) {
                    self.consume_with_pat(place, *pat);
                }
            }
            Expr::UnaryOp { expr, op: _ }
            | Expr::Array(Array::Repeat { initializer: expr, repeat: _ })
            | Expr::Await { expr }
            | Expr::Loop { body: expr, label: _ }
            | Expr::Box { expr }
            | Expr::Cast { expr, type_ref: _ } => {
                self.consume_expr(*expr);
            }
            Expr::Ref { expr, rawness: _, mutability } => {
                // We need to do this before we push the span so the order will be correct.
                let place = self.place_of_expr(*expr);
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                match mutability {
                    hir_def::type_ref::Mutability::Shared => self.ref_expr(*expr, place),
                    hir_def::type_ref::Mutability::Mut => self.mutate_expr(*expr, place),
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => {
                let Some(op) = op else {
                    return;
                };
                if matches!(op, BinaryOp::Assignment { .. }) {
                    let place = self.place_of_expr(*lhs);
                    self.mutate_expr(*lhs, place);
                    self.consume_expr(*rhs);
                    return;
                }
                self.consume_expr(*lhs);
                self.consume_expr(*rhs);
            }
            Expr::Range { lhs, rhs, range_type: _ } => {
                if let &Some(expr) = lhs {
                    self.consume_expr(expr);
                }
                if let &Some(expr) = rhs {
                    self.consume_expr(expr);
                }
            }
            Expr::Index { base, index } => {
                self.select_from_expr(*base);
                self.consume_expr(*index);
            }
            Expr::Closure { .. } => {
                let ty = self.expr_ty(tgt_expr);
                let TyKind::Closure(id, _) = ty.kind() else {
                    never!("closure type is always closure");
                    return;
                };
                let (captures, _) =
                    self.result.closure_info.get(&id.0).expect(
                        "We sort closures, so we should always have data for inner closures",
                    );
                let mut cc = mem::take(&mut self.current_captures);
                cc.extend(captures.iter().filter(|it| self.is_upvar(&it.place)).map(|it| {
                    CapturedItemWithoutTy {
                        place: it.place.clone(),
                        kind: it.kind,
                        span_stacks: it.span_stacks.clone(),
                    }
                }));
                self.current_captures = cc;
            }
            Expr::Array(Array::ElementList { elements: exprs }) | Expr::Tuple { exprs } => {
                self.consume_exprs(exprs.iter().copied())
            }
            &Expr::Assignment { target, value } => {
                self.walk_expr(value);
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                match self.place_of_expr(value) {
                    Some(rhs_place) => {
                        self.inside_assignment = true;
                        self.consume_with_pat(rhs_place, target);
                        self.inside_assignment = false;
                    }
                    None => self.body.walk_pats(target, &mut |pat| match &self.body[pat] {
                        Pat::Path(path) => self.mutate_path_pat(path, pat),
                        &Pat::Expr(expr) => {
                            let place = self.place_of_expr(expr);
                            self.mutate_expr(expr, place);
                        }
                        _ => {}
                    }),
                }
                self.resolver.reset_to_guard(resolver_guard);
            }

            Expr::Missing
            | Expr::Continue { .. }
            | Expr::Path(_)
            | Expr::Literal(_)
            | Expr::Const(_)
            | Expr::Underscore => (),
        }
    }

    fn walk_pat(&mut self, result: &mut Option<CaptureKind>, pat: PatId) {
        let mut update_result = |ck: CaptureKind| match result {
            Some(r) => {
                *r = cmp::max(*r, ck);
            }
            None => *result = Some(ck),
        };

        self.walk_pat_inner(
            pat,
            &mut update_result,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
        );
    }

    fn walk_pat_inner(
        &mut self,
        p: PatId,
        update_result: &mut impl FnMut(CaptureKind),
        mut for_mut: BorrowKind,
    ) {
        match &self.body[p] {
            Pat::Ref { .. }
            | Pat::Box { .. }
            | Pat::Missing
            | Pat::Wild
            | Pat::Tuple { .. }
            | Pat::Expr(_)
            | Pat::Or(_) => (),
            Pat::TupleStruct { .. } | Pat::Record { .. } => {
                if let Some(variant) = self.result.variant_resolution_for_pat(p) {
                    let adt = variant.adt_id(self.db);
                    let is_multivariant = match adt {
                        hir_def::AdtId::EnumId(e) => e.enum_variants(self.db).variants.len() != 1,
                        _ => false,
                    };
                    if is_multivariant {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    }
                }
            }
            Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Path(_)
            | Pat::Lit(_)
            | Pat::Range { .. } => {
                update_result(CaptureKind::ByRef(BorrowKind::Shared));
            }
            Pat::Bind { id, .. } => match self.result.binding_modes[p] {
                crate::BindingMode::Move => {
                    if self.is_ty_copy(self.result.binding_ty(*id)) {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    } else {
                        update_result(CaptureKind::ByValue);
                    }
                }
                crate::BindingMode::Ref(r) => match r {
                    Mutability::Mut => update_result(CaptureKind::ByRef(for_mut)),
                    Mutability::Not => update_result(CaptureKind::ByRef(BorrowKind::Shared)),
                },
            },
        }
        if self.result.pat_adjustments.get(&p).is_some_and(|it| !it.is_empty()) {
            for_mut = BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture };
        }
        self.body.walk_pats_shallow(p, |p| self.walk_pat_inner(p, update_result, for_mut));
    }

    fn is_upvar(&self, place: &HirPlace) -> bool {
        if let Some(c) = self.current_closure {
            let InternedClosure(_, root) = self.db.lookup_intern_closure(c);
            return self.body.is_binding_upvar(place.local, root);
        }
        false
    }

    fn is_ty_copy(&mut self, ty: Ty<'db>) -> bool {
        if let TyKind::Closure(id, _) = ty.kind() {
            // FIXME: We handle closure as a special case, since chalk consider every closure as copy. We
            // should probably let chalk know which closures are copy, but I don't know how doing it
            // without creating query cycles.
            return self
                .result
                .closure_info
                .get(&id.0)
                .map(|it| it.1 == FnTrait::Fn)
                .unwrap_or(true);
        }
        let ty = self.table.resolve_completely(ty);
        self.table.type_is_copy_modulo_regions(ty)
    }

    fn select_from_expr(&mut self, expr: ExprId) {
        self.walk_expr(expr);
    }

    fn restrict_precision_for_unsafe(&mut self) {
        // FIXME: Borrow checker problems without this.
        let mut current_captures = std::mem::take(&mut self.current_captures);
        for capture in &mut current_captures {
            let mut ty = self.table.resolve_completely(self.result.binding_ty(capture.place.local));
            if ty.is_raw_ptr() || ty.is_union() {
                capture.kind = CaptureKind::ByRef(BorrowKind::Shared);
                self.truncate_capture_spans(capture, 0);
                capture.place.projections.truncate(0);
                continue;
            }
            for (i, p) in capture.place.projections.iter().enumerate() {
                ty = p.projected_ty(
                    &self.table.infer_ctxt,
                    self.table.param_env,
                    ty,
                    self.owner.module(self.db).krate(self.db),
                );
                if ty.is_raw_ptr() || ty.is_union() {
                    capture.kind = CaptureKind::ByRef(BorrowKind::Shared);
                    self.truncate_capture_spans(capture, i + 1);
                    capture.place.projections.truncate(i + 1);
                    break;
                }
            }
        }
        self.current_captures = current_captures;
    }

    fn adjust_for_move_closure(&mut self) {
        // FIXME: Borrow checker won't allow without this.
        let mut current_captures = std::mem::take(&mut self.current_captures);
        for capture in &mut current_captures {
            if let Some(first_deref) =
                capture.place.projections.iter().position(|proj| *proj == HirPlaceProjection::Deref)
            {
                self.truncate_capture_spans(capture, first_deref);
                capture.place.projections.truncate(first_deref);
            }
            capture.kind = CaptureKind::ByValue;
        }
        self.current_captures = current_captures;
    }

    fn minimize_captures(&mut self) {
        self.current_captures.sort_unstable_by_key(|it| it.place.projections.len());
        let mut hash_map = FxHashMap::<HirPlace, usize>::default();
        let result = mem::take(&mut self.current_captures);
        for mut item in result {
            let mut lookup_place = HirPlace { local: item.place.local, projections: vec![] };
            let mut it = item.place.projections.iter();
            let prev_index = loop {
                if let Some(k) = hash_map.get(&lookup_place) {
                    break Some(*k);
                }
                match it.next() {
                    Some(it) => {
                        lookup_place.projections.push(*it);
                    }
                    None => break None,
                }
            };
            match prev_index {
                Some(p) => {
                    let prev_projections_len = self.current_captures[p].place.projections.len();
                    self.truncate_capture_spans(&mut item, prev_projections_len);
                    self.current_captures[p].span_stacks.extend(item.span_stacks);
                    let len = self.current_captures[p].place.projections.len();
                    let kind_after_truncate =
                        item.place.capture_kind_of_truncated_place(item.kind, len);
                    self.current_captures[p].kind =
                        cmp::max(kind_after_truncate, self.current_captures[p].kind);
                }
                None => {
                    hash_map.insert(item.place.clone(), self.current_captures.len());
                    self.current_captures.push(item);
                }
            }
        }
    }

    fn consume_with_pat(&mut self, mut place: HirPlace, tgt_pat: PatId) {
        let adjustments_count =
            self.result.pat_adjustments.get(&tgt_pat).map(|it| it.len()).unwrap_or_default();
        place.projections.extend((0..adjustments_count).map(|_| HirPlaceProjection::Deref));
        self.current_capture_span_stack
            .extend((0..adjustments_count).map(|_| MirSpan::PatId(tgt_pat)));
        'reset_span_stack: {
            match &self.body[tgt_pat] {
                Pat::Missing | Pat::Wild => (),
                Pat::Tuple { args, ellipsis } => {
                    let (al, ar) = args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
                    let field_count = match self.result.pat_ty(tgt_pat).kind() {
                        TyKind::Tuple(s) => s.len(),
                        _ => break 'reset_span_stack,
                    };
                    let fields = 0..field_count;
                    let it = al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                    for (&arg, i) in it {
                        let mut p = place.clone();
                        self.current_capture_span_stack.push(MirSpan::PatId(arg));
                        p.projections.push(HirPlaceProjection::TupleField(i as u32));
                        self.consume_with_pat(p, arg);
                        self.current_capture_span_stack.pop();
                    }
                }
                Pat::Or(pats) => {
                    for pat in pats.iter() {
                        self.consume_with_pat(place.clone(), *pat);
                    }
                }
                Pat::Record { args, .. } => {
                    let Some(variant) = self.result.variant_resolution_for_pat(tgt_pat) else {
                        break 'reset_span_stack;
                    };
                    match variant {
                        VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                            self.consume_place(place)
                        }
                        VariantId::StructId(s) => {
                            let vd = s.fields(self.db);
                            for field_pat in args.iter() {
                                let arg = field_pat.pat;
                                let Some(local_id) = vd.field(&field_pat.name) else {
                                    continue;
                                };
                                let mut p = place.clone();
                                self.current_capture_span_stack.push(MirSpan::PatId(arg));
                                p.projections.push(HirPlaceProjection::Field(FieldId {
                                    parent: variant,
                                    local_id,
                                }));
                                self.consume_with_pat(p, arg);
                                self.current_capture_span_stack.pop();
                            }
                        }
                    }
                }
                Pat::Range { .. } | Pat::Slice { .. } | Pat::ConstBlock(_) | Pat::Lit(_) => {
                    self.consume_place(place)
                }
                Pat::Path(path) => {
                    if self.inside_assignment {
                        self.mutate_path_pat(path, tgt_pat);
                    }
                    self.consume_place(place);
                }
                &Pat::Bind { id, subpat: _ } => {
                    let mode = self.result.binding_modes[tgt_pat];
                    let capture_kind = match mode {
                        BindingMode::Move => {
                            self.consume_place(place);
                            break 'reset_span_stack;
                        }
                        BindingMode::Ref(Mutability::Not) => BorrowKind::Shared,
                        BindingMode::Ref(Mutability::Mut) => {
                            BorrowKind::Mut { kind: MutBorrowKind::Default }
                        }
                    };
                    self.current_capture_span_stack.push(MirSpan::BindingId(id));
                    self.add_capture(place, CaptureKind::ByRef(capture_kind));
                    self.current_capture_span_stack.pop();
                }
                Pat::TupleStruct { path: _, args, ellipsis } => {
                    let Some(variant) = self.result.variant_resolution_for_pat(tgt_pat) else {
                        break 'reset_span_stack;
                    };
                    match variant {
                        VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                            self.consume_place(place)
                        }
                        VariantId::StructId(s) => {
                            let vd = s.fields(self.db);
                            let (al, ar) =
                                args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
                            let fields = vd.fields().iter();
                            let it = al
                                .iter()
                                .zip(fields.clone())
                                .chain(ar.iter().rev().zip(fields.rev()));
                            for (&arg, (i, _)) in it {
                                let mut p = place.clone();
                                self.current_capture_span_stack.push(MirSpan::PatId(arg));
                                p.projections.push(HirPlaceProjection::Field(FieldId {
                                    parent: variant,
                                    local_id: i,
                                }));
                                self.consume_with_pat(p, arg);
                                self.current_capture_span_stack.pop();
                            }
                        }
                    }
                }
                Pat::Ref { pat, mutability: _ } => {
                    self.current_capture_span_stack.push(MirSpan::PatId(tgt_pat));
                    place.projections.push(HirPlaceProjection::Deref);
                    self.consume_with_pat(place, *pat);
                    self.current_capture_span_stack.pop();
                }
                Pat::Box { .. } => (), // not supported
                &Pat::Expr(expr) => {
                    self.consume_place(place);
                    let pat_capture_span_stack = mem::take(&mut self.current_capture_span_stack);
                    let old_inside_assignment = mem::replace(&mut self.inside_assignment, false);
                    let lhs_place = self.place_of_expr(expr);
                    self.mutate_expr(expr, lhs_place);
                    self.inside_assignment = old_inside_assignment;
                    self.current_capture_span_stack = pat_capture_span_stack;
                }
            }
        }
        self.current_capture_span_stack
            .truncate(self.current_capture_span_stack.len() - adjustments_count);
    }

    fn consume_exprs(&mut self, exprs: impl Iterator<Item = ExprId>) {
        for expr in exprs {
            self.consume_expr(expr);
        }
    }

    fn closure_kind(&self) -> FnTrait {
        let mut r = FnTrait::Fn;
        for it in &self.current_captures {
            r = cmp::min(
                r,
                match &it.kind {
                    CaptureKind::ByRef(BorrowKind::Mut { .. }) => FnTrait::FnMut,
                    CaptureKind::ByRef(BorrowKind::Shallow | BorrowKind::Shared) => FnTrait::Fn,
                    CaptureKind::ByValue => FnTrait::FnOnce,
                },
            )
        }
        r
    }

    fn analyze_closure(&mut self, closure: InternedClosureId) -> FnTrait {
        let InternedClosure(_, root) = self.db.lookup_intern_closure(closure);
        self.current_closure = Some(closure);
        let Expr::Closure { body, capture_by, .. } = &self.body[root] else {
            unreachable!("Closure expression id is always closure");
        };
        self.consume_expr(*body);
        for item in &self.current_captures {
            if matches!(
                item.kind,
                CaptureKind::ByRef(BorrowKind::Mut {
                    kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow
                })
            ) && !item.place.projections.contains(&HirPlaceProjection::Deref)
            {
                // FIXME: remove the `mutated_bindings_in_closure` completely and add proper fake reads in
                // MIR. I didn't do that due duplicate diagnostics.
                self.result.mutated_bindings_in_closure.insert(item.place.local);
            }
        }
        self.restrict_precision_for_unsafe();
        // `closure_kind` should be done before adjust_for_move_closure
        // If there exists pre-deduced kind of a closure, use it instead of one determined by capture, as rustc does.
        // rustc also does diagnostics here if the latter is not a subtype of the former.
        let closure_kind = self
            .result
            .closure_info
            .get(&closure)
            .map_or_else(|| self.closure_kind(), |info| info.1);
        match capture_by {
            CaptureBy::Value => self.adjust_for_move_closure(),
            CaptureBy::Ref => (),
        }
        self.minimize_captures();
        self.strip_captures_ref_span();
        let result = mem::take(&mut self.current_captures);
        let captures = result.into_iter().map(|it| it.with_ty(self)).collect::<Vec<_>>();
        self.result.closure_info.insert(closure, (captures, closure_kind));
        closure_kind
    }

    fn strip_captures_ref_span(&mut self) {
        // FIXME: Borrow checker won't allow without this.
        let mut captures = std::mem::take(&mut self.current_captures);
        for capture in &mut captures {
            if matches!(capture.kind, CaptureKind::ByValue) {
                for span_stack in &mut capture.span_stacks {
                    if span_stack[span_stack.len() - 1].is_ref_span(self.body) {
                        span_stack.truncate(span_stack.len() - 1);
                    }
                }
            }
        }
        self.current_captures = captures;
    }

    pub(crate) fn infer_closures(&mut self) {
        let deferred_closures = self.sort_closures();
        for (closure, exprs) in deferred_closures.into_iter().rev() {
            self.current_captures = vec![];
            let kind = self.analyze_closure(closure);

            for (derefed_callee, callee_ty, params, expr) in exprs {
                if let &Expr::Call { callee, .. } = &self.body[expr] {
                    let mut adjustments =
                        self.result.expr_adjustments.remove(&callee).unwrap_or_default().into_vec();
                    self.write_fn_trait_method_resolution(
                        kind,
                        derefed_callee,
                        &mut adjustments,
                        callee_ty,
                        &params,
                        expr,
                    );
                    self.result.expr_adjustments.insert(callee, adjustments.into_boxed_slice());
                }
            }
        }
    }

    /// We want to analyze some closures before others, to have a correct analysis:
    /// * We should analyze nested closures before the parent, since the parent should capture some of
    ///   the things that its children captures.
    /// * If a closure calls another closure, we need to analyze the callee, to find out how we should
    ///   capture it (e.g. by move for FnOnce)
    ///
    /// These dependencies are collected in the main inference. We do a topological sort in this function. It
    /// will consume the `deferred_closures` field and return its content in a sorted vector.
    fn sort_closures(
        &mut self,
    ) -> Vec<(InternedClosureId, Vec<(Ty<'db>, Ty<'db>, Vec<Ty<'db>>, ExprId)>)> {
        let mut deferred_closures = mem::take(&mut self.deferred_closures);
        let mut dependents_count: FxHashMap<InternedClosureId, usize> =
            deferred_closures.keys().map(|it| (*it, 0)).collect();
        for deps in self.closure_dependencies.values() {
            for dep in deps {
                *dependents_count.entry(*dep).or_default() += 1;
            }
        }
        let mut queue: Vec<_> =
            deferred_closures.keys().copied().filter(|&it| dependents_count[&it] == 0).collect();
        let mut result = vec![];
        while let Some(it) = queue.pop() {
            if let Some(d) = deferred_closures.remove(&it) {
                result.push((it, d));
            }
            for &dep in self.closure_dependencies.get(&it).into_iter().flat_map(|it| it.iter()) {
                let cnt = dependents_count.get_mut(&dep).unwrap();
                *cnt -= 1;
                if *cnt == 0 {
                    queue.push(dep);
                }
            }
        }
        assert!(deferred_closures.is_empty(), "we should have analyzed all closures");
        result
    }

    pub(crate) fn add_current_closure_dependency(&mut self, dep: InternedClosureId) {
        if let Some(c) = self.current_closure
            && !dep_creates_cycle(&self.closure_dependencies, &mut FxHashSet::default(), c, dep)
        {
            self.closure_dependencies.entry(c).or_default().push(dep);
        }

        fn dep_creates_cycle(
            closure_dependencies: &FxHashMap<InternedClosureId, Vec<InternedClosureId>>,
            visited: &mut FxHashSet<InternedClosureId>,
            from: InternedClosureId,
            to: InternedClosureId,
        ) -> bool {
            if !visited.insert(from) {
                return false;
            }

            if from == to {
                return true;
            }

            if let Some(deps) = closure_dependencies.get(&to) {
                for dep in deps {
                    if dep_creates_cycle(closure_dependencies, visited, from, *dep) {
                        return true;
                    }
                }
            }

            false
        }
    }
}

/// Call this only when the last span in the stack isn't a split.
fn apply_adjusts_to_place(
    current_capture_span_stack: &mut Vec<MirSpan>,
    mut r: HirPlace,
    adjustments: &[Adjustment],
) -> Option<HirPlace> {
    let span = *current_capture_span_stack.last().expect("empty capture span stack");
    for adj in adjustments {
        match &adj.kind {
            Adjust::Deref(None) => {
                current_capture_span_stack.push(span);
                r.projections.push(HirPlaceProjection::Deref);
            }
            _ => return None,
        }
    }
    Some(r)
}
