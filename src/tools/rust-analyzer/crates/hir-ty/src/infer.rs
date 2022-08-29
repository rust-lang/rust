//! Type inference, i.e. the process of walking through the code and determining
//! the type of each expression and pattern.
//!
//! For type inference, compare the implementations in rustc (the various
//! check_* methods in librustc_typeck/check/mod.rs are a good entry point) and
//! IntelliJ-Rust (org.rust.lang.core.types.infer). Our entry point for
//! inference here is the `infer` function, which infers the types of all
//! expressions in a given function.
//!
//! During inference, types (i.e. the `Ty` struct) can contain type 'variables'
//! which represent currently unknown types; as we walk through the expressions,
//! we might determine that certain variables need to be equal to each other, or
//! to certain types. To record this, we use the union-find implementation from
//! the `ena` crate, which is extracted from rustc.

use std::ops::Index;
use std::sync::Arc;

use chalk_ir::{cast::Cast, ConstValue, DebruijnIndex, Mutability, Safety, Scalar, TypeFlags};
use hir_def::{
    body::Body,
    data::{ConstData, StaticData},
    expr::{BindingAnnotation, ExprId, PatId},
    lang_item::LangItemTarget,
    path::{path, Path},
    resolver::{HasResolver, ResolveValueResult, Resolver, TypeNs, ValueNs},
    type_ref::TypeRef,
    AdtId, AssocItemId, DefWithBodyId, EnumVariantId, FieldId, FunctionId, HasModule, Lookup,
    TraitId, TypeAliasId, VariantId,
};
use hir_expand::name::{name, Name};
use itertools::Either;
use la_arena::ArenaMap;
use rustc_hash::FxHashMap;
use stdx::{always, impl_from};

use crate::{
    db::HirDatabase, fold_tys, fold_tys_and_consts, infer::coerce::CoerceMany,
    lower::ImplTraitLoweringMode, to_assoc_type_id, AliasEq, AliasTy, Const, DomainGoal,
    GenericArg, Goal, ImplTraitId, InEnvironment, Interner, ProjectionTy, Substitution,
    TraitEnvironment, TraitRef, Ty, TyBuilder, TyExt, TyKind,
};

// This lint has a false positive here. See the link below for details.
//
// https://github.com/rust-lang/rust/issues/57411
#[allow(unreachable_pub)]
pub use coerce::could_coerce;
#[allow(unreachable_pub)]
pub use unify::could_unify;

pub(crate) mod unify;
mod path;
mod expr;
mod pat;
mod coerce;
mod closure;

/// The entry point of type inference.
pub(crate) fn infer_query(db: &dyn HirDatabase, def: DefWithBodyId) -> Arc<InferenceResult> {
    let _p = profile::span("infer_query");
    let resolver = def.resolver(db.upcast());
    let body = db.body(def);
    let mut ctx = InferenceContext::new(db, def, &body, resolver);

    match def {
        DefWithBodyId::ConstId(c) => ctx.collect_const(&db.const_data(c)),
        DefWithBodyId::FunctionId(f) => ctx.collect_fn(f),
        DefWithBodyId::StaticId(s) => ctx.collect_static(&db.static_data(s)),
    }

    ctx.infer_body();

    Arc::new(ctx.resolve_all())
}

/// Fully normalize all the types found within `ty` in context of `owner` body definition.
///
/// This is appropriate to use only after type-check: it assumes
/// that normalization will succeed, for example.
pub(crate) fn normalize(db: &dyn HirDatabase, owner: DefWithBodyId, ty: Ty) -> Ty {
    if !ty.data(Interner).flags.intersects(TypeFlags::HAS_PROJECTION) {
        return ty;
    }
    let krate = owner.module(db.upcast()).krate();
    let trait_env = owner
        .as_generic_def_id()
        .map_or_else(|| Arc::new(TraitEnvironment::empty(krate)), |d| db.trait_environment(d));
    let mut table = unify::InferenceTable::new(db, trait_env);

    let ty_with_vars = table.normalize_associated_types_in(ty);
    table.resolve_obligations_as_possible();
    table.propagate_diverging_flag();
    table.resolve_completely(ty_with_vars)
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
enum ExprOrPatId {
    ExprId(ExprId),
    PatId(PatId),
}
impl_from!(ExprId, PatId for ExprOrPatId);

/// Binding modes inferred for patterns.
/// <https://doc.rust-lang.org/reference/patterns.html#binding-modes>
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BindingMode {
    Move,
    Ref(Mutability),
}

impl BindingMode {
    fn convert(annotation: BindingAnnotation) -> BindingMode {
        match annotation {
            BindingAnnotation::Unannotated | BindingAnnotation::Mutable => BindingMode::Move,
            BindingAnnotation::Ref => BindingMode::Ref(Mutability::Not),
            BindingAnnotation::RefMut => BindingMode::Ref(Mutability::Mut),
        }
    }
}

impl Default for BindingMode {
    fn default() -> Self {
        BindingMode::Move
    }
}

/// Used to generalize patterns and assignee expressions.
trait PatLike: Into<ExprOrPatId> + Copy {
    type BindingMode: Copy;

    fn infer(
        this: &mut InferenceContext<'_>,
        id: Self,
        expected_ty: &Ty,
        default_bm: Self::BindingMode,
    ) -> Ty;
}

impl PatLike for ExprId {
    type BindingMode = ();

    fn infer(
        this: &mut InferenceContext<'_>,
        id: Self,
        expected_ty: &Ty,
        _: Self::BindingMode,
    ) -> Ty {
        this.infer_assignee_expr(id, expected_ty)
    }
}

impl PatLike for PatId {
    type BindingMode = BindingMode;

    fn infer(
        this: &mut InferenceContext<'_>,
        id: Self,
        expected_ty: &Ty,
        default_bm: Self::BindingMode,
    ) -> Ty {
        this.infer_pat(id, expected_ty, default_bm)
    }
}

#[derive(Debug)]
pub(crate) struct InferOk<T> {
    value: T,
    goals: Vec<InEnvironment<Goal>>,
}

impl<T> InferOk<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> InferOk<U> {
        InferOk { value: f(self.value), goals: self.goals }
    }
}

#[derive(Debug)]
pub(crate) struct TypeError;
pub(crate) type InferResult<T> = Result<InferOk<T>, TypeError>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum InferenceDiagnostic {
    NoSuchField { expr: ExprId },
    BreakOutsideOfLoop { expr: ExprId },
    MismatchedArgCount { call_expr: ExprId, expected: usize, found: usize },
}

/// A mismatch between an expected and an inferred type.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypeMismatch {
    pub expected: Ty,
    pub actual: Ty,
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct InternedStandardTypes {
    unknown: Ty,
    bool_: Ty,
    unit: Ty,
}

impl Default for InternedStandardTypes {
    fn default() -> Self {
        InternedStandardTypes {
            unknown: TyKind::Error.intern(Interner),
            bool_: TyKind::Scalar(Scalar::Bool).intern(Interner),
            unit: TyKind::Tuple(0, Substitution::empty(Interner)).intern(Interner),
        }
    }
}
/// Represents coercing a value to a different type of value.
///
/// We transform values by following a number of `Adjust` steps in order.
/// See the documentation on variants of `Adjust` for more details.
///
/// Here are some common scenarios:
///
/// 1. The simplest cases are where a pointer is not adjusted fat vs thin.
///    Here the pointer will be dereferenced N times (where a dereference can
///    happen to raw or borrowed pointers or any smart pointer which implements
///    Deref, including Box<_>). The types of dereferences is given by
///    `autoderefs`. It can then be auto-referenced zero or one times, indicated
///    by `autoref`, to either a raw or borrowed pointer. In these cases unsize is
///    `false`.
///
/// 2. A thin-to-fat coercion involves unsizing the underlying data. We start
///    with a thin pointer, deref a number of times, unsize the underlying data,
///    then autoref. The 'unsize' phase may change a fixed length array to a
///    dynamically sized one, a concrete object to a trait object, or statically
///    sized struct to a dynamically sized one. E.g., &[i32; 4] -> &[i32] is
///    represented by:
///
///    ```
///    Deref(None) -> [i32; 4],
///    Borrow(AutoBorrow::Ref) -> &[i32; 4],
///    Unsize -> &[i32],
///    ```
///
///    Note that for a struct, the 'deep' unsizing of the struct is not recorded.
///    E.g., `struct Foo<T> { x: T }` we can coerce &Foo<[i32; 4]> to &Foo<[i32]>
///    The autoderef and -ref are the same as in the above example, but the type
///    stored in `unsize` is `Foo<[i32]>`, we don't store any further detail about
///    the underlying conversions from `[i32; 4]` to `[i32]`.
///
/// 3. Coercing a `Box<T>` to `Box<dyn Trait>` is an interesting special case. In
///    that case, we have the pointer we need coming in, so there are no
///    autoderefs, and no autoref. Instead we just do the `Unsize` transformation.
///    At some point, of course, `Box` should move out of the compiler, in which
///    case this is analogous to transforming a struct. E.g., Box<[i32; 4]> ->
///    Box<[i32]> is an `Adjust::Unsize` with the target `Box<[i32]>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Adjustment {
    pub kind: Adjust,
    pub target: Ty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Adjust {
    /// Go from ! to any type.
    NeverToAny,
    /// Dereference once, producing a place.
    Deref(Option<OverloadedDeref>),
    /// Take the address and produce either a `&` or `*` pointer.
    Borrow(AutoBorrow),
    Pointer(PointerCast),
}

/// An overloaded autoderef step, representing a `Deref(Mut)::deref(_mut)`
/// call, with the signature `&'a T -> &'a U` or `&'a mut T -> &'a mut U`.
/// The target type is `U` in both cases, with the region and mutability
/// being those shared by both the receiver and the returned reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverloadedDeref(pub Mutability);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AutoBorrow {
    /// Converts from T to &T.
    Ref(Mutability),
    /// Converts from T to *T.
    RawPtr(Mutability),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerCast {
    /// Go from a fn-item type to a fn-pointer type.
    ReifyFnPointer,

    /// Go from a safe fn pointer to an unsafe fn pointer.
    UnsafeFnPointer,

    /// Go from a non-capturing closure to an fn pointer or an unsafe fn pointer.
    /// It cannot convert a closure that requires unsafe.
    ClosureFnPointer(Safety),

    /// Go from a mut raw pointer to a const raw pointer.
    MutToConstPointer,

    #[allow(dead_code)]
    /// Go from `*const [T; N]` to `*const T`
    ArrayToPointer,

    /// Unsize a pointer/reference value, e.g., `&[T; n]` to
    /// `&[T]`. Note that the source could be a thin or fat pointer.
    /// This will do things like convert thin pointers to fat
    /// pointers, or convert structs containing thin pointers to
    /// structs containing fat pointers, or convert between fat
    /// pointers. We don't store the details of how the transform is
    /// done (in fact, we don't know that, because it might depend on
    /// the precise type parameters). We just store the target
    /// type. Codegen backends and miri figure out what has to be done
    /// based on the precise source/target type at hand.
    Unsize,
}

/// The result of type inference: A mapping from expressions and patterns to types.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct InferenceResult {
    /// For each method call expr, records the function it resolves to.
    method_resolutions: FxHashMap<ExprId, (FunctionId, Substitution)>,
    /// For each field access expr, records the field it resolves to.
    field_resolutions: FxHashMap<ExprId, FieldId>,
    /// For each struct literal or pattern, records the variant it resolves to.
    variant_resolutions: FxHashMap<ExprOrPatId, VariantId>,
    /// For each associated item record what it resolves to
    assoc_resolutions: FxHashMap<ExprOrPatId, AssocItemId>,
    pub diagnostics: Vec<InferenceDiagnostic>,
    pub type_of_expr: ArenaMap<ExprId, Ty>,
    /// For each pattern record the type it resolves to.
    ///
    /// **Note**: When a pattern type is resolved it may still contain
    /// unresolved or missing subpatterns or subpatterns of mismatched types.
    pub type_of_pat: ArenaMap<PatId, Ty>,
    type_mismatches: FxHashMap<ExprOrPatId, TypeMismatch>,
    /// Interned Unknown to return references to.
    standard_types: InternedStandardTypes,
    /// Stores the types which were implicitly dereferenced in pattern binding modes.
    pub pat_adjustments: FxHashMap<PatId, Vec<Ty>>,
    pub pat_binding_modes: FxHashMap<PatId, BindingMode>,
    pub expr_adjustments: FxHashMap<ExprId, Vec<Adjustment>>,
}

impl InferenceResult {
    pub fn method_resolution(&self, expr: ExprId) -> Option<(FunctionId, Substitution)> {
        self.method_resolutions.get(&expr).cloned()
    }
    pub fn field_resolution(&self, expr: ExprId) -> Option<FieldId> {
        self.field_resolutions.get(&expr).copied()
    }
    pub fn variant_resolution_for_expr(&self, id: ExprId) -> Option<VariantId> {
        self.variant_resolutions.get(&id.into()).copied()
    }
    pub fn variant_resolution_for_pat(&self, id: PatId) -> Option<VariantId> {
        self.variant_resolutions.get(&id.into()).copied()
    }
    pub fn assoc_resolutions_for_expr(&self, id: ExprId) -> Option<AssocItemId> {
        self.assoc_resolutions.get(&id.into()).copied()
    }
    pub fn assoc_resolutions_for_pat(&self, id: PatId) -> Option<AssocItemId> {
        self.assoc_resolutions.get(&id.into()).copied()
    }
    pub fn type_mismatch_for_expr(&self, expr: ExprId) -> Option<&TypeMismatch> {
        self.type_mismatches.get(&expr.into())
    }
    pub fn type_mismatch_for_pat(&self, pat: PatId) -> Option<&TypeMismatch> {
        self.type_mismatches.get(&pat.into())
    }
    pub fn expr_type_mismatches(&self) -> impl Iterator<Item = (ExprId, &TypeMismatch)> {
        self.type_mismatches.iter().filter_map(|(expr_or_pat, mismatch)| match *expr_or_pat {
            ExprOrPatId::ExprId(expr) => Some((expr, mismatch)),
            _ => None,
        })
    }
    pub fn pat_type_mismatches(&self) -> impl Iterator<Item = (PatId, &TypeMismatch)> {
        self.type_mismatches.iter().filter_map(|(expr_or_pat, mismatch)| match *expr_or_pat {
            ExprOrPatId::PatId(pat) => Some((pat, mismatch)),
            _ => None,
        })
    }
}

impl Index<ExprId> for InferenceResult {
    type Output = Ty;

    fn index(&self, expr: ExprId) -> &Ty {
        self.type_of_expr.get(expr).unwrap_or(&self.standard_types.unknown)
    }
}

impl Index<PatId> for InferenceResult {
    type Output = Ty;

    fn index(&self, pat: PatId) -> &Ty {
        self.type_of_pat.get(pat).unwrap_or(&self.standard_types.unknown)
    }
}

/// The inference context contains all information needed during type inference.
#[derive(Clone, Debug)]
pub(crate) struct InferenceContext<'a> {
    pub(crate) db: &'a dyn HirDatabase,
    pub(crate) owner: DefWithBodyId,
    pub(crate) body: &'a Body,
    pub(crate) resolver: Resolver,
    table: unify::InferenceTable<'a>,
    trait_env: Arc<TraitEnvironment>,
    pub(crate) result: InferenceResult,
    /// The return type of the function being inferred, the closure or async block if we're
    /// currently within one.
    ///
    /// We might consider using a nested inference context for checking
    /// closures, but currently this is the only field that will change there,
    /// so it doesn't make sense.
    return_ty: Ty,
    diverges: Diverges,
    breakables: Vec<BreakableContext>,
}

#[derive(Clone, Debug)]
struct BreakableContext {
    may_break: bool,
    coerce: CoerceMany,
    label: Option<name::Name>,
}

fn find_breakable<'c>(
    ctxs: &'c mut [BreakableContext],
    label: Option<&name::Name>,
) -> Option<&'c mut BreakableContext> {
    match label {
        Some(_) => ctxs.iter_mut().rev().find(|ctx| ctx.label.as_ref() == label),
        None => ctxs.last_mut(),
    }
}

impl<'a> InferenceContext<'a> {
    fn new(
        db: &'a dyn HirDatabase,
        owner: DefWithBodyId,
        body: &'a Body,
        resolver: Resolver,
    ) -> Self {
        let krate = owner.module(db.upcast()).krate();
        let trait_env = owner
            .as_generic_def_id()
            .map_or_else(|| Arc::new(TraitEnvironment::empty(krate)), |d| db.trait_environment(d));
        InferenceContext {
            result: InferenceResult::default(),
            table: unify::InferenceTable::new(db, trait_env.clone()),
            trait_env,
            return_ty: TyKind::Error.intern(Interner), // set in collect_fn_signature
            db,
            owner,
            body,
            resolver,
            diverges: Diverges::Maybe,
            breakables: Vec::new(),
        }
    }

    fn resolve_all(self) -> InferenceResult {
        let InferenceContext { mut table, mut result, .. } = self;

        // FIXME resolve obligations as well (use Guidance if necessary)
        table.resolve_obligations_as_possible();

        // make sure diverging type variables are marked as such
        table.propagate_diverging_flag();
        for ty in result.type_of_expr.values_mut() {
            *ty = table.resolve_completely(ty.clone());
        }
        for ty in result.type_of_pat.values_mut() {
            *ty = table.resolve_completely(ty.clone());
        }
        for mismatch in result.type_mismatches.values_mut() {
            mismatch.expected = table.resolve_completely(mismatch.expected.clone());
            mismatch.actual = table.resolve_completely(mismatch.actual.clone());
        }
        for (_, subst) in result.method_resolutions.values_mut() {
            *subst = table.resolve_completely(subst.clone());
        }
        for adjustment in result.expr_adjustments.values_mut().flatten() {
            adjustment.target = table.resolve_completely(adjustment.target.clone());
        }
        for adjustment in result.pat_adjustments.values_mut().flatten() {
            *adjustment = table.resolve_completely(adjustment.clone());
        }
        result
    }

    fn collect_const(&mut self, data: &ConstData) {
        self.return_ty = self.make_ty(&data.type_ref);
    }

    fn collect_static(&mut self, data: &StaticData) {
        self.return_ty = self.make_ty(&data.type_ref);
    }

    fn collect_fn(&mut self, func: FunctionId) {
        let data = self.db.function_data(func);
        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver)
            .with_impl_trait_mode(ImplTraitLoweringMode::Param);
        let param_tys =
            data.params.iter().map(|(_, type_ref)| ctx.lower_ty(type_ref)).collect::<Vec<_>>();
        for (ty, pat) in param_tys.into_iter().zip(self.body.params.iter()) {
            let ty = self.insert_type_vars(ty);
            let ty = self.normalize_associated_types_in(ty);

            self.infer_pat(*pat, &ty, BindingMode::default());
        }
        let error_ty = &TypeRef::Error;
        let return_ty = if data.has_async_kw() {
            data.async_ret_type.as_deref().unwrap_or(error_ty)
        } else {
            &*data.ret_type
        };
        let return_ty = self.make_ty_with_mode(return_ty, ImplTraitLoweringMode::Opaque);
        self.return_ty = return_ty;

        if let Some(rpits) = self.db.return_type_impl_traits(func) {
            // RPIT opaque types use substitution of their parent function.
            let fn_placeholders = TyBuilder::placeholder_subst(self.db, func);
            self.return_ty = fold_tys(
                self.return_ty.clone(),
                |ty, _| {
                    let opaque_ty_id = match ty.kind(Interner) {
                        TyKind::OpaqueType(opaque_ty_id, _) => *opaque_ty_id,
                        _ => return ty,
                    };
                    let idx = match self.db.lookup_intern_impl_trait_id(opaque_ty_id.into()) {
                        ImplTraitId::ReturnTypeImplTrait(_, idx) => idx,
                        _ => unreachable!(),
                    };
                    let bounds = (*rpits).map_ref(|rpits| {
                        rpits.impl_traits[idx as usize].bounds.map_ref(|it| it.into_iter())
                    });
                    let var = self.table.new_type_var();
                    let var_subst = Substitution::from1(Interner, var.clone());
                    for bound in bounds {
                        let predicate =
                            bound.map(|it| it.cloned()).substitute(Interner, &fn_placeholders);
                        let (var_predicate, binders) = predicate
                            .substitute(Interner, &var_subst)
                            .into_value_and_skipped_binders();
                        always!(binders.len(Interner) == 0); // quantified where clauses not yet handled
                        self.push_obligation(var_predicate.cast(Interner));
                    }
                    var
                },
                DebruijnIndex::INNERMOST,
            );
        }
    }

    fn infer_body(&mut self) {
        self.infer_expr_coerce(self.body.body_expr, &Expectation::has_type(self.return_ty.clone()));
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty) {
        self.result.type_of_expr.insert(expr, ty);
    }

    fn write_expr_adj(&mut self, expr: ExprId, adjustments: Vec<Adjustment>) {
        self.result.expr_adjustments.insert(expr, adjustments);
    }

    fn write_method_resolution(&mut self, expr: ExprId, func: FunctionId, subst: Substitution) {
        self.result.method_resolutions.insert(expr, (func, subst));
    }

    fn write_variant_resolution(&mut self, id: ExprOrPatId, variant: VariantId) {
        self.result.variant_resolutions.insert(id, variant);
    }

    fn write_assoc_resolution(&mut self, id: ExprOrPatId, item: AssocItemId) {
        self.result.assoc_resolutions.insert(id, item);
    }

    fn write_pat_ty(&mut self, pat: PatId, ty: Ty) {
        self.result.type_of_pat.insert(pat, ty);
    }

    fn push_diagnostic(&mut self, diagnostic: InferenceDiagnostic) {
        self.result.diagnostics.push(diagnostic);
    }

    fn make_ty_with_mode(
        &mut self,
        type_ref: &TypeRef,
        impl_trait_mode: ImplTraitLoweringMode,
    ) -> Ty {
        // FIXME use right resolver for block
        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver)
            .with_impl_trait_mode(impl_trait_mode);
        let ty = ctx.lower_ty(type_ref);
        let ty = self.insert_type_vars(ty);
        self.normalize_associated_types_in(ty)
    }

    fn make_ty(&mut self, type_ref: &TypeRef) -> Ty {
        self.make_ty_with_mode(type_ref, ImplTraitLoweringMode::Disallowed)
    }

    fn err_ty(&self) -> Ty {
        self.result.standard_types.unknown.clone()
    }

    /// Replaces ConstScalar::Unknown by a new type var, so we can maybe still infer it.
    fn insert_const_vars_shallow(&mut self, c: Const) -> Const {
        let data = c.data(Interner);
        match data.value {
            ConstValue::Concrete(cc) => match cc.interned {
                hir_def::type_ref::ConstScalar::Unknown => {
                    self.table.new_const_var(data.ty.clone())
                }
                _ => c,
            },
            _ => c,
        }
    }

    /// Replaces Ty::Unknown by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty.kind(Interner) {
            TyKind::Error => self.table.new_type_var(),
            TyKind::InferenceVar(..) => {
                let ty_resolved = self.resolve_ty_shallow(&ty);
                if ty_resolved.is_unknown() {
                    self.table.new_type_var()
                } else {
                    ty
                }
            }
            _ => ty,
        }
    }

    fn insert_type_vars(&mut self, ty: Ty) -> Ty {
        fold_tys_and_consts(
            ty,
            |x, _| match x {
                Either::Left(ty) => Either::Left(self.insert_type_vars_shallow(ty)),
                Either::Right(c) => Either::Right(self.insert_const_vars_shallow(c)),
            },
            DebruijnIndex::INNERMOST,
        )
    }

    fn resolve_obligations_as_possible(&mut self) {
        self.table.resolve_obligations_as_possible();
    }

    fn push_obligation(&mut self, o: DomainGoal) {
        self.table.register_obligation(o.cast(Interner));
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        self.table.unify(ty1, ty2)
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    fn normalize_associated_types_in(&mut self, ty: Ty) -> Ty {
        self.table.normalize_associated_types_in(ty)
    }

    fn resolve_ty_shallow(&mut self, ty: &Ty) -> Ty {
        self.resolve_obligations_as_possible();
        self.table.resolve_ty_shallow(ty)
    }

    fn resolve_associated_type(&mut self, inner_ty: Ty, assoc_ty: Option<TypeAliasId>) -> Ty {
        self.resolve_associated_type_with_params(inner_ty, assoc_ty, &[])
    }

    fn resolve_associated_type_with_params(
        &mut self,
        inner_ty: Ty,
        assoc_ty: Option<TypeAliasId>,
        params: &[GenericArg],
    ) -> Ty {
        match assoc_ty {
            Some(res_assoc_ty) => {
                let trait_ = match res_assoc_ty.lookup(self.db.upcast()).container {
                    hir_def::ItemContainerId::TraitId(trait_) => trait_,
                    _ => panic!("resolve_associated_type called with non-associated type"),
                };
                let ty = self.table.new_type_var();
                let mut param_iter = params.iter().cloned();
                let trait_ref = TyBuilder::trait_ref(self.db, trait_)
                    .push(inner_ty)
                    .fill(|_| param_iter.next().unwrap())
                    .build();
                let alias_eq = AliasEq {
                    alias: AliasTy::Projection(ProjectionTy {
                        associated_ty_id: to_assoc_type_id(res_assoc_ty),
                        substitution: trait_ref.substitution.clone(),
                    }),
                    ty: ty.clone(),
                };
                self.push_obligation(trait_ref.cast(Interner));
                self.push_obligation(alias_eq.cast(Interner));
                ty
            }
            None => self.err_ty(),
        }
    }

    fn resolve_variant(&mut self, path: Option<&Path>, value_ns: bool) -> (Ty, Option<VariantId>) {
        let path = match path {
            Some(path) => path,
            None => return (self.err_ty(), None),
        };
        let resolver = &self.resolver;
        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
        // FIXME: this should resolve assoc items as well, see this example:
        // https://play.rust-lang.org/?gist=087992e9e22495446c01c0d4e2d69521
        let (resolution, unresolved) = if value_ns {
            match resolver.resolve_path_in_value_ns(self.db.upcast(), path.mod_path()) {
                Some(ResolveValueResult::ValueNs(value)) => match value {
                    ValueNs::EnumVariantId(var) => {
                        let substs = ctx.substs_from_path(path, var.into(), true);
                        let ty = self.db.ty(var.parent.into());
                        let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                        return (ty, Some(var.into()));
                    }
                    ValueNs::StructId(strukt) => {
                        let substs = ctx.substs_from_path(path, strukt.into(), true);
                        let ty = self.db.ty(strukt.into());
                        let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                        return (ty, Some(strukt.into()));
                    }
                    ValueNs::ImplSelf(impl_id) => (TypeNs::SelfType(impl_id), None),
                    _ => return (self.err_ty(), None),
                },
                Some(ResolveValueResult::Partial(typens, unresolved)) => (typens, Some(unresolved)),
                None => return (self.err_ty(), None),
            }
        } else {
            match resolver.resolve_path_in_type_ns(self.db.upcast(), path.mod_path()) {
                Some(it) => it,
                None => return (self.err_ty(), None),
            }
        };
        return match resolution {
            TypeNs::AdtId(AdtId::StructId(strukt)) => {
                let substs = ctx.substs_from_path(path, strukt.into(), true);
                let ty = self.db.ty(strukt.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                forbid_unresolved_segments((ty, Some(strukt.into())), unresolved)
            }
            TypeNs::AdtId(AdtId::UnionId(u)) => {
                let substs = ctx.substs_from_path(path, u.into(), true);
                let ty = self.db.ty(u.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                forbid_unresolved_segments((ty, Some(u.into())), unresolved)
            }
            TypeNs::EnumVariantId(var) => {
                let substs = ctx.substs_from_path(path, var.into(), true);
                let ty = self.db.ty(var.parent.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                forbid_unresolved_segments((ty, Some(var.into())), unresolved)
            }
            TypeNs::SelfType(impl_id) => {
                let generics = crate::utils::generics(self.db.upcast(), impl_id.into());
                let substs = generics.placeholder_subst(self.db);
                let ty = self.db.impl_self_ty(impl_id).substitute(Interner, &substs);
                self.resolve_variant_on_alias(ty, unresolved, path)
            }
            TypeNs::TypeAliasId(it) => {
                let ty = TyBuilder::def_ty(self.db, it.into())
                    .fill_with_inference_vars(&mut self.table)
                    .build();
                self.resolve_variant_on_alias(ty, unresolved, path)
            }
            TypeNs::AdtSelfType(_) => {
                // FIXME this could happen in array size expressions, once we're checking them
                (self.err_ty(), None)
            }
            TypeNs::GenericParam(_) => {
                // FIXME potentially resolve assoc type
                (self.err_ty(), None)
            }
            TypeNs::AdtId(AdtId::EnumId(_)) | TypeNs::BuiltinType(_) | TypeNs::TraitId(_) => {
                // FIXME diagnostic
                (self.err_ty(), None)
            }
        };

        fn forbid_unresolved_segments(
            result: (Ty, Option<VariantId>),
            unresolved: Option<usize>,
        ) -> (Ty, Option<VariantId>) {
            if unresolved.is_none() {
                result
            } else {
                // FIXME diagnostic
                (TyKind::Error.intern(Interner), None)
            }
        }
    }

    fn resolve_variant_on_alias(
        &mut self,
        ty: Ty,
        unresolved: Option<usize>,
        path: &Path,
    ) -> (Ty, Option<VariantId>) {
        let remaining = unresolved.map(|x| path.segments().skip(x).len()).filter(|x| x > &0);
        match remaining {
            None => {
                let variant = ty.as_adt().and_then(|(adt_id, _)| match adt_id {
                    AdtId::StructId(s) => Some(VariantId::StructId(s)),
                    AdtId::UnionId(u) => Some(VariantId::UnionId(u)),
                    AdtId::EnumId(_) => {
                        // FIXME Error E0071, expected struct, variant or union type, found enum `Foo`
                        None
                    }
                });
                (ty, variant)
            }
            Some(1) => {
                let segment = path.mod_path().segments().last().unwrap();
                // this could be an enum variant or associated type
                if let Some((AdtId::EnumId(enum_id), _)) = ty.as_adt() {
                    let enum_data = self.db.enum_data(enum_id);
                    if let Some(local_id) = enum_data.variant(segment) {
                        let variant = EnumVariantId { parent: enum_id, local_id };
                        return (ty, Some(variant.into()));
                    }
                }
                // FIXME potentially resolve assoc type
                (self.err_ty(), None)
            }
            Some(_) => {
                // FIXME diagnostic
                (self.err_ty(), None)
            }
        }
    }

    fn resolve_lang_item(&self, name: Name) -> Option<LangItemTarget> {
        let krate = self.resolver.krate();
        self.db.lang_item(krate, name.to_smol_str())
    }

    fn resolve_into_iter_item(&self) -> Option<TypeAliasId> {
        let path = path![core::iter::IntoIterator];
        let trait_ = self.resolver.resolve_known_trait(self.db.upcast(), &path)?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Item])
    }

    fn resolve_ops_try_ok(&self) -> Option<TypeAliasId> {
        // FIXME resolve via lang_item once try v2 is stable
        let path = path![core::ops::Try];
        let trait_ = self.resolver.resolve_known_trait(self.db.upcast(), &path)?;
        let trait_data = self.db.trait_data(trait_);
        trait_data
            // FIXME remove once try v2 is stable
            .associated_type_by_name(&name![Ok])
            .or_else(|| trait_data.associated_type_by_name(&name![Output]))
    }

    fn resolve_ops_neg_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_lang_item(name![neg])?.as_trait()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }

    fn resolve_ops_not_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_lang_item(name![not])?.as_trait()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }

    fn resolve_future_future_output(&self) -> Option<TypeAliasId> {
        let trait_ = self
            .resolver
            .resolve_known_trait(self.db.upcast(), &path![core::future::IntoFuture])
            .or_else(|| self.resolve_lang_item(name![future_trait])?.as_trait())?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }

    fn resolve_boxed_box(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(name![owned_box])?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_full(&self) -> Option<AdtId> {
        let path = path![core::ops::RangeFull];
        let struct_ = self.resolver.resolve_known_struct(self.db.upcast(), &path)?;
        Some(struct_.into())
    }

    fn resolve_range(&self) -> Option<AdtId> {
        let path = path![core::ops::Range];
        let struct_ = self.resolver.resolve_known_struct(self.db.upcast(), &path)?;
        Some(struct_.into())
    }

    fn resolve_range_inclusive(&self) -> Option<AdtId> {
        let path = path![core::ops::RangeInclusive];
        let struct_ = self.resolver.resolve_known_struct(self.db.upcast(), &path)?;
        Some(struct_.into())
    }

    fn resolve_range_from(&self) -> Option<AdtId> {
        let path = path![core::ops::RangeFrom];
        let struct_ = self.resolver.resolve_known_struct(self.db.upcast(), &path)?;
        Some(struct_.into())
    }

    fn resolve_range_to(&self) -> Option<AdtId> {
        let path = path![core::ops::RangeTo];
        let struct_ = self.resolver.resolve_known_struct(self.db.upcast(), &path)?;
        Some(struct_.into())
    }

    fn resolve_range_to_inclusive(&self) -> Option<AdtId> {
        let path = path![core::ops::RangeToInclusive];
        let struct_ = self.resolver.resolve_known_struct(self.db.upcast(), &path)?;
        Some(struct_.into())
    }

    fn resolve_ops_index(&self) -> Option<TraitId> {
        self.resolve_lang_item(name![index])?.as_trait()
    }

    fn resolve_ops_index_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_ops_index()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }
}

/// When inferring an expression, we propagate downward whatever type hint we
/// are able in the form of an `Expectation`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Expectation {
    None,
    HasType(Ty),
    // Castable(Ty), // rustc has this, we currently just don't propagate an expectation for casts
    RValueLikeUnsized(Ty),
}

impl Expectation {
    /// The expectation that the type of the expression needs to equal the given
    /// type.
    fn has_type(ty: Ty) -> Self {
        if ty.is_unknown() {
            // FIXME: get rid of this?
            Expectation::None
        } else {
            Expectation::HasType(ty)
        }
    }

    fn from_option(ty: Option<Ty>) -> Self {
        ty.map_or(Expectation::None, Expectation::HasType)
    }

    /// The following explanation is copied straight from rustc:
    /// Provides an expectation for an rvalue expression given an *optional*
    /// hint, which is not required for type safety (the resulting type might
    /// be checked higher up, as is the case with `&expr` and `box expr`), but
    /// is useful in determining the concrete type.
    ///
    /// The primary use case is where the expected type is a fat pointer,
    /// like `&[isize]`. For example, consider the following statement:
    ///
    ///    let x: &[isize] = &[1, 2, 3];
    ///
    /// In this case, the expected type for the `&[1, 2, 3]` expression is
    /// `&[isize]`. If however we were to say that `[1, 2, 3]` has the
    /// expectation `ExpectHasType([isize])`, that would be too strong --
    /// `[1, 2, 3]` does not have the type `[isize]` but rather `[isize; 3]`.
    /// It is only the `&[1, 2, 3]` expression as a whole that can be coerced
    /// to the type `&[isize]`. Therefore, we propagate this more limited hint,
    /// which still is useful, because it informs integer literals and the like.
    /// See the test case `test/ui/coerce-expect-unsized.rs` and #20169
    /// for examples of where this comes up,.
    fn rvalue_hint(table: &mut unify::InferenceTable<'_>, ty: Ty) -> Self {
        // FIXME: do struct_tail_without_normalization
        match table.resolve_ty_shallow(&ty).kind(Interner) {
            TyKind::Slice(_) | TyKind::Str | TyKind::Dyn(_) => Expectation::RValueLikeUnsized(ty),
            _ => Expectation::has_type(ty),
        }
    }

    /// This expresses no expectation on the type.
    fn none() -> Self {
        Expectation::None
    }

    fn resolve(&self, table: &mut unify::InferenceTable<'_>) -> Expectation {
        match self {
            Expectation::None => Expectation::None,
            Expectation::HasType(t) => Expectation::HasType(table.resolve_ty_shallow(t)),
            Expectation::RValueLikeUnsized(t) => {
                Expectation::RValueLikeUnsized(table.resolve_ty_shallow(t))
            }
        }
    }

    fn to_option(&self, table: &mut unify::InferenceTable<'_>) -> Option<Ty> {
        match self.resolve(table) {
            Expectation::None => None,
            Expectation::HasType(t) |
            // Expectation::Castable(t) |
            Expectation::RValueLikeUnsized(t) => Some(t),
        }
    }

    fn only_has_type(&self, table: &mut unify::InferenceTable<'_>) -> Option<Ty> {
        match self {
            Expectation::HasType(t) => Some(table.resolve_ty_shallow(t)),
            // Expectation::Castable(_) |
            Expectation::RValueLikeUnsized(_) | Expectation::None => None,
        }
    }

    /// Comment copied from rustc:
    /// Disregard "castable to" expectations because they
    /// can lead us astray. Consider for example `if cond
    /// {22} else {c} as u8` -- if we propagate the
    /// "castable to u8" constraint to 22, it will pick the
    /// type 22u8, which is overly constrained (c might not
    /// be a u8). In effect, the problem is that the
    /// "castable to" expectation is not the tightest thing
    /// we can say, so we want to drop it in this case.
    /// The tightest thing we can say is "must unify with
    /// else branch". Note that in the case of a "has type"
    /// constraint, this limitation does not hold.
    ///
    /// If the expected type is just a type variable, then don't use
    /// an expected type. Otherwise, we might write parts of the type
    /// when checking the 'then' block which are incompatible with the
    /// 'else' branch.
    fn adjust_for_branches(&self, table: &mut unify::InferenceTable<'_>) -> Expectation {
        match self {
            Expectation::HasType(ety) => {
                let ety = table.resolve_ty_shallow(ety);
                if !ety.is_ty_var() {
                    Expectation::HasType(ety)
                } else {
                    Expectation::None
                }
            }
            Expectation::RValueLikeUnsized(ety) => Expectation::RValueLikeUnsized(ety.clone()),
            _ => Expectation::None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Diverges {
    Maybe,
    Always,
}

impl Diverges {
    fn is_always(self) -> bool {
        self == Diverges::Always
    }
}

impl std::ops::BitAnd for Diverges {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        std::cmp::min(self, other)
    }
}

impl std::ops::BitOr for Diverges {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }
}

impl std::ops::BitAndAssign for Diverges {
    fn bitand_assign(&mut self, other: Self) {
        *self = *self & other;
    }
}

impl std::ops::BitOrAssign for Diverges {
    fn bitor_assign(&mut self, other: Self) {
        *self = *self | other;
    }
}
