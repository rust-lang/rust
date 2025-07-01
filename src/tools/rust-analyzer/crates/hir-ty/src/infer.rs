//! Type inference, i.e. the process of walking through the code and determining
//! the type of each expression and pattern.
//!
//! For type inference, compare the implementations in rustc (the various
//! check_* methods in rustc_hir_analysis/check/mod.rs are a good entry point) and
//! IntelliJ-Rust (org.rust.lang.core.types.infer). Our entry point for
//! inference here is the `infer` function, which infers the types of all
//! expressions in a given function.
//!
//! During inference, types (i.e. the `Ty` struct) can contain type 'variables'
//! which represent currently unknown types; as we walk through the expressions,
//! we might determine that certain variables need to be equal to each other, or
//! to certain types. To record this, we use the union-find implementation from
//! the `ena` crate, which is extracted from rustc.

pub(crate) mod cast;
pub(crate) mod closure;
mod coerce;
pub(crate) mod diagnostics;
mod expr;
mod mutability;
mod pat;
mod path;
pub(crate) mod unify;

use std::{cell::OnceCell, convert::identity, iter, ops::Index};

use chalk_ir::{
    DebruijnIndex, Mutability, Safety, Scalar, TyKind, TypeFlags, Variance,
    cast::Cast,
    fold::TypeFoldable,
    interner::HasInterner,
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
};
use either::Either;
use hir_def::{
    AdtId, AssocItemId, ConstId, DefWithBodyId, FieldId, FunctionId, GenericDefId, GenericParamId,
    ImplId, ItemContainerId, LocalFieldId, Lookup, TraitId, TupleFieldId, TupleId, TypeAliasId,
    VariantId,
    builtin_type::{BuiltinInt, BuiltinType, BuiltinUint},
    expr_store::{Body, ExpressionStore, HygieneId, path::Path},
    hir::{BindingAnnotation, BindingId, ExprId, ExprOrPatId, LabelId, PatId},
    lang_item::{LangItem, LangItemTarget, lang_item},
    layout::Integer,
    resolver::{HasResolver, ResolveValueResult, Resolver, TypeNs, ValueNs},
    signatures::{ConstSignature, StaticSignature},
    type_ref::{ConstRef, LifetimeRefId, TypeRefId},
};
use hir_expand::{mod_path::ModPath, name::Name};
use indexmap::IndexSet;
use intern::sym;
use la_arena::{ArenaMap, Entry};
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::{always, never};
use triomphe::Arc;

use crate::{
    AliasEq, AliasTy, Binders, ClosureId, Const, DomainGoal, GenericArg, Goal, ImplTraitId,
    ImplTraitIdx, InEnvironment, IncorrectGenericsLenKind, Interner, Lifetime, OpaqueTyId,
    ParamLoweringMode, PathLoweringDiagnostic, ProjectionTy, Substitution, TraitEnvironment, Ty,
    TyBuilder, TyExt,
    db::HirDatabase,
    fold_tys,
    generics::Generics,
    infer::{
        coerce::CoerceMany,
        diagnostics::{Diagnostics, InferenceTyLoweringContext as TyLoweringContext},
        expr::ExprIsRead,
        unify::InferenceTable,
    },
    lower::{ImplTraitLoweringMode, LifetimeElisionKind, diagnostics::TyLoweringDiagnostic},
    mir::MirSpan,
    static_lifetime, to_assoc_type_id,
    traits::FnTrait,
    utils::UnevaluatedConstEvaluatorFolder,
};

// This lint has a false positive here. See the link below for details.
//
// https://github.com/rust-lang/rust/issues/57411
#[allow(unreachable_pub)]
pub use coerce::could_coerce;
#[allow(unreachable_pub)]
pub use unify::{could_unify, could_unify_deeply};

use cast::{CastCheck, CastError};
pub(crate) use closure::{CaptureKind, CapturedItem, CapturedItemWithoutTy};

/// The entry point of type inference.
pub(crate) fn infer_query(db: &dyn HirDatabase, def: DefWithBodyId) -> Arc<InferenceResult> {
    let _p = tracing::info_span!("infer_query").entered();
    let resolver = def.resolver(db);
    let body = db.body(def);
    let mut ctx = InferenceContext::new(db, def, &body, resolver);

    match def {
        DefWithBodyId::FunctionId(f) => {
            ctx.collect_fn(f);
        }
        DefWithBodyId::ConstId(c) => ctx.collect_const(c, &db.const_signature(c)),
        DefWithBodyId::StaticId(s) => ctx.collect_static(&db.static_signature(s)),
        DefWithBodyId::VariantId(v) => {
            ctx.return_ty = TyBuilder::builtin(
                match db.enum_signature(v.lookup(db).parent).variant_body_type() {
                    hir_def::layout::IntegerType::Pointer(signed) => match signed {
                        true => BuiltinType::Int(BuiltinInt::Isize),
                        false => BuiltinType::Uint(BuiltinUint::Usize),
                    },
                    hir_def::layout::IntegerType::Fixed(size, signed) => match signed {
                        true => BuiltinType::Int(match size {
                            Integer::I8 => BuiltinInt::I8,
                            Integer::I16 => BuiltinInt::I16,
                            Integer::I32 => BuiltinInt::I32,
                            Integer::I64 => BuiltinInt::I64,
                            Integer::I128 => BuiltinInt::I128,
                        }),
                        false => BuiltinType::Uint(match size {
                            Integer::I8 => BuiltinUint::U8,
                            Integer::I16 => BuiltinUint::U16,
                            Integer::I32 => BuiltinUint::U32,
                            Integer::I64 => BuiltinUint::U64,
                            Integer::I128 => BuiltinUint::U128,
                        }),
                    },
                },
            );
        }
    }

    ctx.infer_body();

    ctx.infer_mut_body();

    ctx.infer_closures();

    Arc::new(ctx.resolve_all())
}

pub(crate) fn infer_cycle_result(_: &dyn HirDatabase, _: DefWithBodyId) -> Arc<InferenceResult> {
    Arc::new(InferenceResult { has_errors: true, ..Default::default() })
}

/// Fully normalize all the types found within `ty` in context of `owner` body definition.
///
/// This is appropriate to use only after type-check: it assumes
/// that normalization will succeed, for example.
pub(crate) fn normalize(db: &dyn HirDatabase, trait_env: Arc<TraitEnvironment>, ty: Ty) -> Ty {
    // FIXME: TypeFlags::HAS_CT_PROJECTION is not implemented in chalk, so TypeFlags::HAS_PROJECTION only
    // works for the type case, so we check array unconditionally. Remove the array part
    // when the bug in chalk becomes fixed.
    if !ty.data(Interner).flags.intersects(TypeFlags::HAS_PROJECTION)
        && !matches!(ty.kind(Interner), TyKind::Array(..))
    {
        return ty;
    }
    let mut table = unify::InferenceTable::new(db, trait_env);

    let ty_with_vars = table.normalize_associated_types_in(ty);
    table.resolve_obligations_as_possible();
    table.propagate_diverging_flag();
    table.resolve_completely(ty_with_vars)
}

/// Binding modes inferred for patterns.
/// <https://doc.rust-lang.org/reference/patterns.html#binding-modes>
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum BindingMode {
    #[default]
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InferenceTyDiagnosticSource {
    /// Diagnostics that come from types in the body.
    Body,
    /// Diagnostics that come from types in fn parameters/return type, or static & const types.
    Signature,
}

#[derive(Debug)]
pub(crate) struct TypeError;
pub(crate) type InferResult<T> = Result<InferOk<T>, TypeError>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum InferenceDiagnostic {
    NoSuchField {
        field: ExprOrPatId,
        private: Option<LocalFieldId>,
        variant: VariantId,
    },
    PrivateField {
        expr: ExprId,
        field: FieldId,
    },
    PrivateAssocItem {
        id: ExprOrPatId,
        item: AssocItemId,
    },
    UnresolvedField {
        expr: ExprId,
        receiver: Ty,
        name: Name,
        method_with_same_name_exists: bool,
    },
    UnresolvedMethodCall {
        expr: ExprId,
        receiver: Ty,
        name: Name,
        /// Contains the type the field resolves to
        field_with_same_name: Option<Ty>,
        assoc_func_with_same_name: Option<FunctionId>,
    },
    UnresolvedAssocItem {
        id: ExprOrPatId,
    },
    UnresolvedIdent {
        id: ExprOrPatId,
    },
    // FIXME: This should be emitted in body lowering
    BreakOutsideOfLoop {
        expr: ExprId,
        is_break: bool,
        bad_value_break: bool,
    },
    MismatchedArgCount {
        call_expr: ExprId,
        expected: usize,
        found: usize,
    },
    MismatchedTupleStructPatArgCount {
        pat: ExprOrPatId,
        expected: usize,
        found: usize,
    },
    ExpectedFunction {
        call_expr: ExprId,
        found: Ty,
    },
    TypedHole {
        expr: ExprId,
        expected: Ty,
    },
    CastToUnsized {
        expr: ExprId,
        cast_ty: Ty,
    },
    InvalidCast {
        expr: ExprId,
        error: CastError,
        expr_ty: Ty,
        cast_ty: Ty,
    },
    TyDiagnostic {
        source: InferenceTyDiagnosticSource,
        diag: TyLoweringDiagnostic,
    },
    PathDiagnostic {
        node: ExprOrPatId,
        diag: PathLoweringDiagnostic,
    },
    MethodCallIncorrectGenericsLen {
        expr: ExprId,
        provided_count: u32,
        expected_count: u32,
        kind: IncorrectGenericsLenKind,
        def: GenericDefId,
    },
    MethodCallIncorrectGenericsOrder {
        expr: ExprId,
        param_id: GenericParamId,
        arg_idx: u32,
        /// Whether the `GenericArgs` contains a `Self` arg.
        has_self_arg: bool,
    },
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
    never: Ty,
}

impl Default for InternedStandardTypes {
    fn default() -> Self {
        InternedStandardTypes {
            unknown: TyKind::Error.intern(Interner),
            bool_: TyKind::Scalar(Scalar::Bool).intern(Interner),
            unit: TyKind::Tuple(0, Substitution::empty(Interner)).intern(Interner),
            never: TyKind::Never.intern(Interner),
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
///    ```ignore
///    Deref(None) -> [i32; 4],
///    Borrow(AutoBorrow::Ref) -> &[i32; 4],
///    Unsize -> &[i32],
///    ```
///
///    Note that for a struct, the 'deep' unsizing of the struct is not recorded.
///    E.g., `struct Foo<T> { it: T }` we can coerce &Foo<[i32; 4]> to &Foo<[i32]>
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

impl Adjustment {
    pub fn borrow(m: Mutability, ty: Ty, lt: Lifetime) -> Self {
        let ty = TyKind::Ref(m, lt.clone(), ty).intern(Interner);
        Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(lt, m)), target: ty }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
///
/// Mutability is `None` when we are not sure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverloadedDeref(pub Option<Mutability>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AutoBorrow {
    /// Converts from T to &T.
    Ref(Lifetime, Mutability),
    /// Converts from T to *T.
    RawPtr(Mutability),
}

impl AutoBorrow {
    fn mutability(&self) -> Mutability {
        let (AutoBorrow::Ref(_, m) | AutoBorrow::RawPtr(m)) = self;
        *m
    }
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
///
/// When you add a field that stores types (including `Substitution` and the like), don't forget
/// `resolve_completely()`'ing  them in `InferenceContext::resolve_all()`. Inference variables must
/// not appear in the final inference result.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct InferenceResult {
    /// For each method call expr, records the function it resolves to.
    method_resolutions: FxHashMap<ExprId, (FunctionId, Substitution)>,
    /// For each field access expr, records the field it resolves to.
    field_resolutions: FxHashMap<ExprId, Either<FieldId, TupleFieldId>>,
    /// For each struct literal or pattern, records the variant it resolves to.
    variant_resolutions: FxHashMap<ExprOrPatId, VariantId>,
    /// For each associated item record what it resolves to
    assoc_resolutions: FxHashMap<ExprOrPatId, (AssocItemId, Substitution)>,
    /// Whenever a tuple field expression access a tuple field, we allocate a tuple id in
    /// [`InferenceContext`] and store the tuples substitution there. This map is the reverse of
    /// that which allows us to resolve a [`TupleFieldId`]s type.
    pub tuple_field_access_types: FxHashMap<TupleId, Substitution>,
    /// During inference this field is empty and [`InferenceContext::diagnostics`] is filled instead.
    pub diagnostics: Vec<InferenceDiagnostic>,
    pub type_of_expr: ArenaMap<ExprId, Ty>,
    /// For each pattern record the type it resolves to.
    ///
    /// **Note**: When a pattern type is resolved it may still contain
    /// unresolved or missing subpatterns or subpatterns of mismatched types.
    pub type_of_pat: ArenaMap<PatId, Ty>,
    pub type_of_binding: ArenaMap<BindingId, Ty>,
    pub type_of_rpit: ArenaMap<ImplTraitIdx, Ty>,
    /// Type of the result of `.into_iter()` on the for. `ExprId` is the one of the whole for loop.
    pub type_of_for_iterator: FxHashMap<ExprId, Ty>,
    type_mismatches: FxHashMap<ExprOrPatId, TypeMismatch>,
    /// Whether there are any type-mismatching errors in the result.
    // FIXME: This isn't as useful as initially thought due to us falling back placeholders to
    // `TyKind::Error`.
    // Which will then mark this field.
    pub(crate) has_errors: bool,
    /// Interned common types to return references to.
    // FIXME: Move this into `InferenceContext`
    standard_types: InternedStandardTypes,
    /// Stores the types which were implicitly dereferenced in pattern binding modes.
    pub pat_adjustments: FxHashMap<PatId, Vec<Ty>>,
    /// Stores the binding mode (`ref` in `let ref x = 2`) of bindings.
    ///
    /// This one is tied to the `PatId` instead of `BindingId`, because in some rare cases, a binding in an
    /// or pattern can have multiple binding modes. For example:
    /// ```
    /// fn foo(mut slice: &[u32]) -> usize {
    ///     slice = match slice {
    ///         [0, rest @ ..] | rest => rest,
    ///     };
    ///     0
    /// }
    /// ```
    /// the first `rest` has implicit `ref` binding mode, but the second `rest` binding mode is `move`.
    pub binding_modes: ArenaMap<PatId, BindingMode>,
    pub expr_adjustments: FxHashMap<ExprId, Box<[Adjustment]>>,
    pub(crate) closure_info: FxHashMap<ClosureId, (Vec<CapturedItem>, FnTrait)>,
    // FIXME: remove this field
    pub mutated_bindings_in_closure: FxHashSet<BindingId>,
    pub coercion_casts: FxHashSet<ExprId>,
}

impl InferenceResult {
    pub fn method_resolution(&self, expr: ExprId) -> Option<(FunctionId, Substitution)> {
        self.method_resolutions.get(&expr).cloned()
    }
    pub fn field_resolution(&self, expr: ExprId) -> Option<Either<FieldId, TupleFieldId>> {
        self.field_resolutions.get(&expr).copied()
    }
    pub fn variant_resolution_for_expr(&self, id: ExprId) -> Option<VariantId> {
        self.variant_resolutions.get(&id.into()).copied()
    }
    pub fn variant_resolution_for_pat(&self, id: PatId) -> Option<VariantId> {
        self.variant_resolutions.get(&id.into()).copied()
    }
    pub fn variant_resolution_for_expr_or_pat(&self, id: ExprOrPatId) -> Option<VariantId> {
        match id {
            ExprOrPatId::ExprId(id) => self.variant_resolution_for_expr(id),
            ExprOrPatId::PatId(id) => self.variant_resolution_for_pat(id),
        }
    }
    pub fn assoc_resolutions_for_expr(&self, id: ExprId) -> Option<(AssocItemId, Substitution)> {
        self.assoc_resolutions.get(&id.into()).cloned()
    }
    pub fn assoc_resolutions_for_pat(&self, id: PatId) -> Option<(AssocItemId, Substitution)> {
        self.assoc_resolutions.get(&id.into()).cloned()
    }
    pub fn assoc_resolutions_for_expr_or_pat(
        &self,
        id: ExprOrPatId,
    ) -> Option<(AssocItemId, Substitution)> {
        match id {
            ExprOrPatId::ExprId(id) => self.assoc_resolutions_for_expr(id),
            ExprOrPatId::PatId(id) => self.assoc_resolutions_for_pat(id),
        }
    }
    pub fn type_mismatch_for_expr(&self, expr: ExprId) -> Option<&TypeMismatch> {
        self.type_mismatches.get(&expr.into())
    }
    pub fn type_mismatch_for_pat(&self, pat: PatId) -> Option<&TypeMismatch> {
        self.type_mismatches.get(&pat.into())
    }
    pub fn type_mismatches(&self) -> impl Iterator<Item = (ExprOrPatId, &TypeMismatch)> {
        self.type_mismatches.iter().map(|(expr_or_pat, mismatch)| (*expr_or_pat, mismatch))
    }
    pub fn expr_type_mismatches(&self) -> impl Iterator<Item = (ExprId, &TypeMismatch)> {
        self.type_mismatches.iter().filter_map(|(expr_or_pat, mismatch)| match *expr_or_pat {
            ExprOrPatId::ExprId(expr) => Some((expr, mismatch)),
            _ => None,
        })
    }
    pub fn closure_info(&self, closure: &ClosureId) -> &(Vec<CapturedItem>, FnTrait) {
        self.closure_info.get(closure).unwrap()
    }
    pub fn type_of_expr_or_pat(&self, id: ExprOrPatId) -> Option<&Ty> {
        match id {
            ExprOrPatId::ExprId(id) => self.type_of_expr.get(id),
            ExprOrPatId::PatId(id) => self.type_of_pat.get(id),
        }
    }
    pub fn is_erroneous(&self) -> bool {
        self.has_errors && self.type_of_expr.iter().count() == 0
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

impl Index<ExprOrPatId> for InferenceResult {
    type Output = Ty;

    fn index(&self, id: ExprOrPatId) -> &Ty {
        self.type_of_expr_or_pat(id).unwrap_or(&self.standard_types.unknown)
    }
}

impl Index<BindingId> for InferenceResult {
    type Output = Ty;

    fn index(&self, b: BindingId) -> &Ty {
        self.type_of_binding.get(b).unwrap_or(&self.standard_types.unknown)
    }
}

/// The inference context contains all information needed during type inference.
#[derive(Clone, Debug)]
pub(crate) struct InferenceContext<'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) owner: DefWithBodyId,
    pub(crate) body: &'db Body,
    /// Generally you should not resolve things via this resolver. Instead create a TyLoweringContext
    /// and resolve the path via its methods. This will ensure proper error reporting.
    pub(crate) resolver: Resolver<'db>,
    generic_def: GenericDefId,
    generics: OnceCell<Generics>,
    table: unify::InferenceTable<'db>,
    /// The traits in scope, disregarding block modules. This is used for caching purposes.
    traits_in_scope: FxHashSet<TraitId>,
    pub(crate) result: InferenceResult,
    tuple_field_accesses_rev:
        IndexSet<Substitution, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>,
    /// The return type of the function being inferred, the closure or async block if we're
    /// currently within one.
    ///
    /// We might consider using a nested inference context for checking
    /// closures so we can swap all shared things out at once.
    return_ty: Ty,
    /// If `Some`, this stores coercion information for returned
    /// expressions. If `None`, this is in a context where return is
    /// inappropriate, such as a const expression.
    return_coercion: Option<CoerceMany>,
    /// The resume type and the yield type, respectively, of the coroutine being inferred.
    resume_yield_tys: Option<(Ty, Ty)>,
    diverges: Diverges,
    breakables: Vec<BreakableContext>,

    /// Whether we are inside the pattern of a destructuring assignment.
    inside_assignment: bool,

    deferred_cast_checks: Vec<CastCheck>,

    // fields related to closure capture
    current_captures: Vec<CapturedItemWithoutTy>,
    /// A stack that has an entry for each projection in the current capture.
    ///
    /// For example, in `a.b.c`, we capture the spans of `a`, `a.b`, and `a.b.c`.
    /// We do that because sometimes we truncate projections (when a closure captures
    /// both `a.b` and `a.b.c`), and we want to provide accurate spans in this case.
    current_capture_span_stack: Vec<MirSpan>,
    current_closure: Option<ClosureId>,
    /// Stores the list of closure ids that need to be analyzed before this closure. See the
    /// comment on `InferenceContext::sort_closures`
    closure_dependencies: FxHashMap<ClosureId, Vec<ClosureId>>,
    deferred_closures: FxHashMap<ClosureId, Vec<(Ty, Ty, Vec<Ty>, ExprId)>>,

    diagnostics: Diagnostics,
}

#[derive(Clone, Debug)]
struct BreakableContext {
    /// Whether this context contains at least one break expression.
    may_break: bool,
    /// The coercion target of the context.
    coerce: Option<CoerceMany>,
    /// The optional label of the context.
    label: Option<LabelId>,
    kind: BreakableKind,
}

#[derive(Clone, Debug)]
enum BreakableKind {
    Block,
    Loop,
    /// A border is something like an async block, closure etc. Anything that prevents
    /// breaking/continuing through
    Border,
}

fn find_breakable(
    ctxs: &mut [BreakableContext],
    label: Option<LabelId>,
) -> Option<&mut BreakableContext> {
    let mut ctxs = ctxs
        .iter_mut()
        .rev()
        .take_while(|it| matches!(it.kind, BreakableKind::Block | BreakableKind::Loop));
    match label {
        Some(_) => ctxs.find(|ctx| ctx.label == label),
        None => ctxs.find(|ctx| matches!(ctx.kind, BreakableKind::Loop)),
    }
}

fn find_continuable(
    ctxs: &mut [BreakableContext],
    label: Option<LabelId>,
) -> Option<&mut BreakableContext> {
    match label {
        Some(_) => find_breakable(ctxs, label).filter(|it| matches!(it.kind, BreakableKind::Loop)),
        None => find_breakable(ctxs, label),
    }
}

enum ImplTraitReplacingMode {
    ReturnPosition(FxHashSet<Ty>),
    TypeAlias,
}

impl<'db> InferenceContext<'db> {
    fn new(
        db: &'db dyn HirDatabase,
        owner: DefWithBodyId,
        body: &'db Body,
        resolver: Resolver<'db>,
    ) -> Self {
        let trait_env = db.trait_environment_for_body(owner);
        InferenceContext {
            generics: OnceCell::new(),
            result: InferenceResult::default(),
            table: unify::InferenceTable::new(db, trait_env),
            tuple_field_accesses_rev: Default::default(),
            return_ty: TyKind::Error.intern(Interner), // set in collect_* calls
            resume_yield_tys: None,
            return_coercion: None,
            db,
            owner,
            generic_def: match owner {
                DefWithBodyId::FunctionId(it) => it.into(),
                DefWithBodyId::StaticId(it) => it.into(),
                DefWithBodyId::ConstId(it) => it.into(),
                DefWithBodyId::VariantId(it) => it.lookup(db).parent.into(),
            },
            body,
            traits_in_scope: resolver.traits_in_scope(db),
            resolver,
            diverges: Diverges::Maybe,
            breakables: Vec::new(),
            deferred_cast_checks: Vec::new(),
            current_captures: Vec::new(),
            current_capture_span_stack: Vec::new(),
            current_closure: None,
            deferred_closures: FxHashMap::default(),
            closure_dependencies: FxHashMap::default(),
            inside_assignment: false,
            diagnostics: Diagnostics::default(),
        }
    }

    pub(crate) fn generics(&self) -> &Generics {
        self.generics.get_or_init(|| crate::generics::generics(self.db, self.generic_def))
    }

    // FIXME: This function should be private in module. It is currently only used in the consteval, since we need
    // `InferenceResult` in the middle of inference. See the fixme comment in `consteval::eval_to_const`. If you
    // used this function for another workaround, mention it here. If you really need this function and believe that
    // there is no problem in it being `pub(crate)`, remove this comment.
    pub(crate) fn resolve_all(self) -> InferenceResult {
        let InferenceContext {
            mut table,
            mut result,
            mut deferred_cast_checks,
            tuple_field_accesses_rev,
            diagnostics,
            ..
        } = self;
        let mut diagnostics = diagnostics.finish();
        // Destructure every single field so whenever new fields are added to `InferenceResult` we
        // don't forget to handle them here.
        let InferenceResult {
            method_resolutions,
            field_resolutions: _,
            variant_resolutions: _,
            assoc_resolutions,
            type_of_expr,
            type_of_pat,
            type_of_binding,
            type_of_rpit,
            type_of_for_iterator,
            type_mismatches,
            has_errors,
            standard_types: _,
            pat_adjustments,
            binding_modes: _,
            expr_adjustments,
            // Types in `closure_info` have already been `resolve_completely()`'d during
            // `InferenceContext::infer_closures()` (in `HirPlace::ty()` specifically), so no need
            // to resolve them here.
            closure_info: _,
            mutated_bindings_in_closure: _,
            tuple_field_access_types: _,
            coercion_casts,
            diagnostics: _,
        } = &mut result;
        table.fallback_if_possible();

        // Comment from rustc:
        // Even though coercion casts provide type hints, we check casts after fallback for
        // backwards compatibility. This makes fallback a stronger type hint than a cast coercion.
        let mut apply_adjustments = |expr, adj: Vec<_>| {
            expr_adjustments.insert(expr, adj.into_boxed_slice());
        };
        let mut set_coercion_cast = |expr| {
            coercion_casts.insert(expr);
        };
        for cast in deferred_cast_checks.iter_mut() {
            if let Err(diag) =
                cast.check(&mut table, &mut apply_adjustments, &mut set_coercion_cast)
            {
                diagnostics.push(diag);
            }
        }

        // FIXME resolve obligations as well (use Guidance if necessary)
        table.resolve_obligations_as_possible();

        // make sure diverging type variables are marked as such
        table.propagate_diverging_flag();
        for ty in type_of_expr.values_mut() {
            *ty = table.resolve_completely(ty.clone());
            *has_errors = *has_errors || ty.contains_unknown();
        }
        type_of_expr.shrink_to_fit();
        for ty in type_of_pat.values_mut() {
            *ty = table.resolve_completely(ty.clone());
            *has_errors = *has_errors || ty.contains_unknown();
        }
        type_of_pat.shrink_to_fit();
        for ty in type_of_binding.values_mut() {
            *ty = table.resolve_completely(ty.clone());
            *has_errors = *has_errors || ty.contains_unknown();
        }
        type_of_binding.shrink_to_fit();
        for ty in type_of_rpit.values_mut() {
            *ty = table.resolve_completely(ty.clone());
            *has_errors = *has_errors || ty.contains_unknown();
        }
        type_of_rpit.shrink_to_fit();
        for ty in type_of_for_iterator.values_mut() {
            *ty = table.resolve_completely(ty.clone());
            *has_errors = *has_errors || ty.contains_unknown();
        }
        type_of_for_iterator.shrink_to_fit();

        *has_errors |= !type_mismatches.is_empty();

        type_mismatches.retain(|_, mismatch| {
            mismatch.expected = table.resolve_completely(mismatch.expected.clone());
            mismatch.actual = table.resolve_completely(mismatch.actual.clone());
            chalk_ir::zip::Zip::zip_with(
                &mut UnknownMismatch(self.db),
                Variance::Invariant,
                &mismatch.expected,
                &mismatch.actual,
            )
            .is_ok()
        });
        type_mismatches.shrink_to_fit();
        diagnostics.retain_mut(|diagnostic| {
            use InferenceDiagnostic::*;
            match diagnostic {
                ExpectedFunction { found: ty, .. }
                | UnresolvedField { receiver: ty, .. }
                | UnresolvedMethodCall { receiver: ty, .. } => {
                    *ty = table.resolve_completely(ty.clone());
                    // FIXME: Remove this when we are on par with rustc in terms of inference
                    if ty.contains_unknown() {
                        return false;
                    }

                    if let UnresolvedMethodCall { field_with_same_name, .. } = diagnostic {
                        if let Some(ty) = field_with_same_name {
                            *ty = table.resolve_completely(ty.clone());
                            if ty.contains_unknown() {
                                *field_with_same_name = None;
                            }
                        }
                    }
                }
                TypedHole { expected: ty, .. } => {
                    *ty = table.resolve_completely(ty.clone());
                }
                _ => (),
            }
            true
        });
        diagnostics.shrink_to_fit();
        for (_, subst) in method_resolutions.values_mut() {
            *subst = table.resolve_completely(subst.clone());
            *has_errors =
                *has_errors || subst.type_parameters(Interner).any(|ty| ty.contains_unknown());
        }
        method_resolutions.shrink_to_fit();
        for (_, subst) in assoc_resolutions.values_mut() {
            *subst = table.resolve_completely(subst.clone());
            *has_errors =
                *has_errors || subst.type_parameters(Interner).any(|ty| ty.contains_unknown());
        }
        assoc_resolutions.shrink_to_fit();
        for adjustment in expr_adjustments.values_mut().flatten() {
            adjustment.target = table.resolve_completely(adjustment.target.clone());
            *has_errors = *has_errors || adjustment.target.contains_unknown();
        }
        expr_adjustments.shrink_to_fit();
        for adjustment in pat_adjustments.values_mut().flatten() {
            *adjustment = table.resolve_completely(adjustment.clone());
            *has_errors = *has_errors || adjustment.contains_unknown();
        }
        pat_adjustments.shrink_to_fit();
        result.tuple_field_access_types = tuple_field_accesses_rev
            .into_iter()
            .enumerate()
            .map(|(idx, subst)| (TupleId(idx as u32), table.resolve_completely(subst)))
            .inspect(|(_, subst)| {
                *has_errors =
                    *has_errors || subst.type_parameters(Interner).any(|ty| ty.contains_unknown());
            })
            .collect();
        result.tuple_field_access_types.shrink_to_fit();

        result.diagnostics = diagnostics;

        result
    }

    fn collect_const(&mut self, id: ConstId, data: &ConstSignature) {
        let return_ty = self.make_ty(
            data.type_ref,
            &data.store,
            InferenceTyDiagnosticSource::Signature,
            LifetimeElisionKind::for_const(id.loc(self.db).container),
        );

        // Constants might be defining usage sites of TAITs.
        self.make_tait_coercion_table(iter::once(&return_ty));

        self.return_ty = return_ty;
    }

    fn collect_static(&mut self, data: &StaticSignature) {
        let return_ty = self.make_ty(
            data.type_ref,
            &data.store,
            InferenceTyDiagnosticSource::Signature,
            LifetimeElisionKind::Elided(static_lifetime()),
        );

        // Statics might be defining usage sites of TAITs.
        self.make_tait_coercion_table(iter::once(&return_ty));

        self.return_ty = return_ty;
    }

    fn collect_fn(&mut self, func: FunctionId) {
        let data = self.db.function_signature(func);
        let mut param_tys = self.with_ty_lowering(
            &data.store,
            InferenceTyDiagnosticSource::Signature,
            LifetimeElisionKind::for_fn_params(&data),
            |ctx| {
                ctx.type_param_mode(ParamLoweringMode::Placeholder);
                data.params.iter().map(|&type_ref| ctx.lower_ty(type_ref)).collect::<Vec<_>>()
            },
        );

        // Check if function contains a va_list, if it does then we append it to the parameter types
        // that are collected from the function data
        if data.is_varargs() {
            let va_list_ty = match self.resolve_va_list() {
                Some(va_list) => TyBuilder::adt(self.db, va_list)
                    .fill_with_defaults(self.db, || self.table.new_type_var())
                    .build(),
                None => self.err_ty(),
            };

            param_tys.push(va_list_ty);
        }
        let mut param_tys = param_tys.into_iter().chain(iter::repeat(self.table.new_type_var()));
        if let Some(self_param) = self.body.self_param {
            if let Some(ty) = param_tys.next() {
                let ty = self.insert_type_vars(ty);
                let ty = self.normalize_associated_types_in(ty);
                self.write_binding_ty(self_param, ty);
            }
        }
        let mut tait_candidates = FxHashSet::default();
        for (ty, pat) in param_tys.zip(&*self.body.params) {
            let ty = self.insert_type_vars(ty);
            let ty = self.normalize_associated_types_in(ty);

            self.infer_top_pat(*pat, &ty, None);
            if ty
                .data(Interner)
                .flags
                .intersects(TypeFlags::HAS_TY_OPAQUE.union(TypeFlags::HAS_TY_INFER))
            {
                tait_candidates.insert(ty);
            }
        }
        let return_ty = match data.ret_type {
            Some(return_ty) => {
                let return_ty = self.with_ty_lowering(
                    &data.store,
                    InferenceTyDiagnosticSource::Signature,
                    LifetimeElisionKind::for_fn_ret(),
                    |ctx| {
                        ctx.type_param_mode(ParamLoweringMode::Placeholder)
                            .impl_trait_mode(ImplTraitLoweringMode::Opaque);
                        ctx.lower_ty(return_ty)
                    },
                );
                let return_ty = self.insert_type_vars(return_ty);
                if let Some(rpits) = self.db.return_type_impl_traits(func) {
                    // RPIT opaque types use substitution of their parent function.
                    let fn_placeholders = TyBuilder::placeholder_subst(self.db, func);
                    let mut mode = ImplTraitReplacingMode::ReturnPosition(FxHashSet::default());
                    let result = self.insert_inference_vars_for_impl_trait(
                        return_ty,
                        fn_placeholders,
                        &mut mode,
                    );
                    if let ImplTraitReplacingMode::ReturnPosition(taits) = mode {
                        tait_candidates.extend(taits);
                    }
                    let rpits = rpits.skip_binders();
                    for (id, _) in rpits.impl_traits.iter() {
                        if let Entry::Vacant(e) = self.result.type_of_rpit.entry(id) {
                            never!("Missed RPIT in `insert_inference_vars_for_rpit`");
                            e.insert(TyKind::Error.intern(Interner));
                        }
                    }
                    result
                } else {
                    return_ty
                }
            }
            None => self.result.standard_types.unit.clone(),
        };

        self.return_ty = self.normalize_associated_types_in(return_ty);
        self.return_coercion = Some(CoerceMany::new(self.return_ty.clone()));

        // Functions might be defining usage sites of TAITs.
        // To define an TAITs, that TAIT must appear in the function's signatures.
        // So, it suffices to check for params and return types.
        if self
            .return_ty
            .data(Interner)
            .flags
            .intersects(TypeFlags::HAS_TY_OPAQUE.union(TypeFlags::HAS_TY_INFER))
        {
            tait_candidates.insert(self.return_ty.clone());
        }
        self.make_tait_coercion_table(tait_candidates.iter());
    }

    fn insert_inference_vars_for_impl_trait<T>(
        &mut self,
        t: T,
        placeholders: Substitution,
        mode: &mut ImplTraitReplacingMode,
    ) -> T
    where
        T: crate::HasInterner<Interner = Interner> + crate::TypeFoldable<Interner>,
    {
        fold_tys(
            t,
            |ty, _| {
                let opaque_ty_id = match ty.kind(Interner) {
                    TyKind::OpaqueType(opaque_ty_id, _) => *opaque_ty_id,
                    _ => return ty,
                };
                let (impl_traits, idx) =
                    match self.db.lookup_intern_impl_trait_id(opaque_ty_id.into()) {
                        // We don't replace opaque types from other kind with inference vars
                        // because `insert_inference_vars_for_impl_traits` for each kinds
                        // and unreplaced opaque types of other kind are resolved while
                        // inferencing because of `tait_coercion_table`.
                        // Moreover, calling `insert_inference_vars_for_impl_traits` with same
                        // `placeholders` for other kind may cause trouble because
                        // the substs for the bounds of each impl traits do not match
                        ImplTraitId::ReturnTypeImplTrait(def, idx) => {
                            if matches!(mode, ImplTraitReplacingMode::TypeAlias) {
                                // RPITs don't have `tait_coercion_table`, so use inserted inference
                                // vars for them.
                                if let Some(ty) = self.result.type_of_rpit.get(idx) {
                                    return ty.clone();
                                }
                                return ty;
                            }
                            (self.db.return_type_impl_traits(def), idx)
                        }
                        ImplTraitId::TypeAliasImplTrait(def, idx) => {
                            if let ImplTraitReplacingMode::ReturnPosition(taits) = mode {
                                // Gather TAITs while replacing RPITs because TAITs inside RPITs
                                // may not visited while replacing TAITs
                                taits.insert(ty.clone());
                                return ty;
                            }
                            (self.db.type_alias_impl_traits(def), idx)
                        }
                        _ => unreachable!(),
                    };
                let Some(impl_traits) = impl_traits else {
                    return ty;
                };
                let bounds = (*impl_traits)
                    .map_ref(|its| its.impl_traits[idx].bounds.map_ref(|it| it.iter()));
                let var = self.table.new_type_var();
                let var_subst = Substitution::from1(Interner, var.clone());
                for bound in bounds {
                    let predicate = bound.map(|it| it.cloned());
                    let predicate = predicate.substitute(Interner, &placeholders);
                    let (var_predicate, binders) =
                        predicate.substitute(Interner, &var_subst).into_value_and_skipped_binders();
                    always!(binders.is_empty(Interner)); // quantified where clauses not yet handled
                    let var_predicate = self.insert_inference_vars_for_impl_trait(
                        var_predicate,
                        placeholders.clone(),
                        mode,
                    );
                    self.push_obligation(var_predicate.cast(Interner));
                }
                self.result.type_of_rpit.insert(idx, var.clone());
                var
            },
            DebruijnIndex::INNERMOST,
        )
    }

    /// The coercion of a non-inference var into an opaque type should fail,
    /// but not in the defining sites of the TAITs.
    /// In such cases, we insert an proxy inference var for each TAIT,
    /// and coerce into it instead of TAIT itself.
    ///
    /// The inference var stretagy is effective because;
    ///
    /// - It can still unify types that coerced into TAITs
    /// - We are pushing `impl Trait` bounds into it
    ///
    /// This function inserts a map that maps the opaque type to that proxy inference var.
    fn make_tait_coercion_table<'b>(&mut self, tait_candidates: impl Iterator<Item = &'b Ty>) {
        struct TypeAliasImplTraitCollector<'a, 'b> {
            db: &'b dyn HirDatabase,
            table: &'b mut InferenceTable<'a>,
            assocs: FxHashMap<OpaqueTyId, (ImplId, Ty)>,
            non_assocs: FxHashMap<OpaqueTyId, Ty>,
        }

        impl TypeVisitor<Interner> for TypeAliasImplTraitCollector<'_, '_> {
            type BreakTy = ();

            fn as_dyn(&mut self) -> &mut dyn TypeVisitor<Interner, BreakTy = Self::BreakTy> {
                self
            }

            fn interner(&self) -> Interner {
                Interner
            }

            fn visit_ty(
                &mut self,
                ty: &chalk_ir::Ty<Interner>,
                outer_binder: DebruijnIndex,
            ) -> std::ops::ControlFlow<Self::BreakTy> {
                let ty = self.table.resolve_ty_shallow(ty);

                if let TyKind::OpaqueType(id, _) = ty.kind(Interner) {
                    if let ImplTraitId::TypeAliasImplTrait(alias_id, _) =
                        self.db.lookup_intern_impl_trait_id((*id).into())
                    {
                        let loc = self.db.lookup_intern_type_alias(alias_id);
                        match loc.container {
                            ItemContainerId::ImplId(impl_id) => {
                                self.assocs.insert(*id, (impl_id, ty.clone()));
                            }
                            ItemContainerId::ModuleId(..) | ItemContainerId::ExternBlockId(..) => {
                                self.non_assocs.insert(*id, ty.clone());
                            }
                            _ => {}
                        }
                    }
                }

                ty.super_visit_with(self, outer_binder)
            }
        }

        let mut collector = TypeAliasImplTraitCollector {
            db: self.db,
            table: &mut self.table,
            assocs: FxHashMap::default(),
            non_assocs: FxHashMap::default(),
        };
        for ty in tait_candidates {
            _ = ty.visit_with(collector.as_dyn(), DebruijnIndex::INNERMOST);
        }

        // Non-assoc TAITs can be define-used everywhere as long as they are
        // in function signatures or const types, etc
        let mut taits = collector.non_assocs;

        // assoc TAITs(ATPITs) can be only define-used inside their impl block.
        // They cannot be define-used in inner items like in the following;
        //
        // ```
        // impl Trait for Struct {
        //     type Assoc = impl Default;
        //
        //     fn assoc_fn() -> Self::Assoc {
        //         let foo: Self::Assoc = true; // Allowed here
        //
        //         fn inner() -> Self::Assoc {
        //              false                   // Not allowed here
        //         }
        //
        //         foo
        //     }
        // }
        // ```
        let impl_id = match self.owner {
            DefWithBodyId::FunctionId(it) => {
                let loc = self.db.lookup_intern_function(it);
                if let ItemContainerId::ImplId(impl_id) = loc.container {
                    Some(impl_id)
                } else {
                    None
                }
            }
            DefWithBodyId::ConstId(it) => {
                let loc = self.db.lookup_intern_const(it);
                if let ItemContainerId::ImplId(impl_id) = loc.container {
                    Some(impl_id)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(impl_id) = impl_id {
            taits.extend(collector.assocs.into_iter().filter_map(|(id, (impl_, ty))| {
                if impl_ == impl_id { Some((id, ty)) } else { None }
            }));
        }

        let tait_coercion_table: FxHashMap<_, _> = taits
            .into_iter()
            .filter_map(|(id, ty)| {
                if let ImplTraitId::TypeAliasImplTrait(alias_id, _) =
                    self.db.lookup_intern_impl_trait_id(id.into())
                {
                    let subst = TyBuilder::placeholder_subst(self.db, alias_id);
                    let ty = self.insert_inference_vars_for_impl_trait(
                        ty,
                        subst,
                        &mut ImplTraitReplacingMode::TypeAlias,
                    );
                    Some((id, ty))
                } else {
                    None
                }
            })
            .collect();

        if !tait_coercion_table.is_empty() {
            self.table.tait_coercion_table = Some(tait_coercion_table);
        }
    }

    fn infer_body(&mut self) {
        match self.return_coercion {
            Some(_) => self.infer_return(self.body.body_expr),
            None => {
                _ = self.infer_expr_coerce(
                    self.body.body_expr,
                    &Expectation::has_type(self.return_ty.clone()),
                    ExprIsRead::Yes,
                )
            }
        }
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty) {
        self.result.type_of_expr.insert(expr, ty);
    }

    fn write_expr_adj(&mut self, expr: ExprId, adjustments: Box<[Adjustment]>) {
        if adjustments.is_empty() {
            return;
        }
        match self.result.expr_adjustments.entry(expr) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                match (&mut entry.get_mut()[..], &adjustments[..]) {
                    (
                        [Adjustment { kind: Adjust::NeverToAny, target }],
                        [.., Adjustment { target: new_target, .. }],
                    ) => {
                        // NeverToAny coercion can target any type, so instead of adding a new
                        // adjustment on top we can change the target.
                        *target = new_target.clone();
                    }
                    _ => {
                        *entry.get_mut() = adjustments;
                    }
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(adjustments);
            }
        }
    }

    fn write_method_resolution(&mut self, expr: ExprId, func: FunctionId, subst: Substitution) {
        self.result.method_resolutions.insert(expr, (func, subst));
    }

    fn write_variant_resolution(&mut self, id: ExprOrPatId, variant: VariantId) {
        self.result.variant_resolutions.insert(id, variant);
    }

    fn write_assoc_resolution(&mut self, id: ExprOrPatId, item: AssocItemId, subs: Substitution) {
        self.result.assoc_resolutions.insert(id, (item, subs));
    }

    fn write_pat_ty(&mut self, pat: PatId, ty: Ty) {
        self.result.type_of_pat.insert(pat, ty);
    }

    fn write_binding_ty(&mut self, id: BindingId, ty: Ty) {
        self.result.type_of_binding.insert(id, ty);
    }

    fn push_diagnostic(&self, diagnostic: InferenceDiagnostic) {
        self.diagnostics.push(diagnostic);
    }

    fn with_ty_lowering<R>(
        &mut self,
        store: &ExpressionStore,
        types_source: InferenceTyDiagnosticSource,
        lifetime_elision: LifetimeElisionKind,
        f: impl FnOnce(&mut TyLoweringContext<'_>) -> R,
    ) -> R {
        let mut ctx = TyLoweringContext::new(
            self.db,
            &self.resolver,
            store,
            &self.diagnostics,
            types_source,
            self.generic_def,
            lifetime_elision,
        );
        f(&mut ctx)
    }

    fn with_body_ty_lowering<R>(&mut self, f: impl FnOnce(&mut TyLoweringContext<'_>) -> R) -> R {
        self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            f,
        )
    }

    fn make_ty(
        &mut self,
        type_ref: TypeRefId,
        store: &ExpressionStore,
        type_source: InferenceTyDiagnosticSource,
        lifetime_elision: LifetimeElisionKind,
    ) -> Ty {
        let ty = self
            .with_ty_lowering(store, type_source, lifetime_elision, |ctx| ctx.lower_ty(type_ref));
        let ty = self.insert_type_vars(ty);
        self.normalize_associated_types_in(ty)
    }

    fn make_body_ty(&mut self, type_ref: TypeRefId) -> Ty {
        self.make_ty(
            type_ref,
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
        )
    }

    fn make_body_const(&mut self, const_ref: ConstRef, ty: Ty) -> Const {
        let const_ = self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            |ctx| {
                ctx.type_param_mode = ParamLoweringMode::Placeholder;
                ctx.lower_const(&const_ref, ty)
            },
        );
        self.insert_type_vars(const_)
    }

    fn make_path_as_body_const(&mut self, path: &Path, ty: Ty) -> Const {
        let const_ = self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            |ctx| {
                ctx.type_param_mode = ParamLoweringMode::Placeholder;
                ctx.lower_path_as_const(path, ty)
            },
        );
        self.insert_type_vars(const_)
    }

    fn err_ty(&self) -> Ty {
        self.result.standard_types.unknown.clone()
    }

    fn make_body_lifetime(&mut self, lifetime_ref: LifetimeRefId) -> Lifetime {
        let lt = self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            |ctx| ctx.lower_lifetime(lifetime_ref),
        );
        self.insert_type_vars(lt)
    }

    /// Replaces `Ty::Error` by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        self.table.insert_type_vars_shallow(ty)
    }

    fn insert_type_vars<T>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        self.table.insert_type_vars(ty)
    }

    fn push_obligation(&mut self, o: DomainGoal) {
        self.table.register_obligation(o.cast(Interner));
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        let ty1 = ty1
            .clone()
            .try_fold_with(
                &mut UnevaluatedConstEvaluatorFolder { db: self.db },
                DebruijnIndex::INNERMOST,
            )
            .unwrap();
        let ty2 = ty2
            .clone()
            .try_fold_with(
                &mut UnevaluatedConstEvaluatorFolder { db: self.db },
                DebruijnIndex::INNERMOST,
            )
            .unwrap();
        self.table.unify(&ty1, &ty2)
    }

    /// Attempts to returns the deeply last field of nested structures, but
    /// does not apply any normalization in its search. Returns the same type
    /// if input `ty` is not a structure at all.
    fn struct_tail_without_normalization(&mut self, ty: Ty) -> Ty {
        self.struct_tail_with_normalize(ty, identity)
    }

    /// Returns the deeply last field of nested structures, or the same type if
    /// not a structure at all. Corresponds to the only possible unsized field,
    /// and its type can be used to determine unsizing strategy.
    ///
    /// This is parameterized over the normalization strategy (i.e. how to
    /// handle `<T as Trait>::Assoc` and `impl Trait`); pass the identity
    /// function to indicate no normalization should take place.
    fn struct_tail_with_normalize(
        &mut self,
        mut ty: Ty,
        mut normalize: impl FnMut(Ty) -> Ty,
    ) -> Ty {
        // FIXME: fetch the limit properly
        let recursion_limit = 10;
        for iteration in 0.. {
            if iteration > recursion_limit {
                return self.err_ty();
            }
            match ty.kind(Interner) {
                TyKind::Adt(chalk_ir::AdtId(hir_def::AdtId::StructId(struct_id)), substs) => {
                    match self.db.field_types((*struct_id).into()).values().next_back().cloned() {
                        Some(field) => {
                            ty = field.substitute(Interner, substs);
                        }
                        None => break,
                    }
                }
                TyKind::Adt(..) => break,
                TyKind::Tuple(_, substs) => {
                    match substs
                        .as_slice(Interner)
                        .split_last()
                        .and_then(|(last_ty, _)| last_ty.ty(Interner))
                    {
                        Some(last_ty) => ty = last_ty.clone(),
                        None => break,
                    }
                }
                TyKind::Alias(..) => {
                    let normalized = normalize(ty.clone());
                    if ty == normalized {
                        return ty;
                    } else {
                        ty = normalized;
                    }
                }
                _ => break,
            }
        }
        ty
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    fn normalize_associated_types_in<T>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        self.table.normalize_associated_types_in(ty)
    }

    fn resolve_ty_shallow(&mut self, ty: &Ty) -> Ty {
        self.table.resolve_ty_shallow(ty)
    }

    fn resolve_associated_type(&mut self, inner_ty: Ty, assoc_ty: Option<TypeAliasId>) -> Ty {
        self.resolve_associated_type_with_params(inner_ty, assoc_ty, &[])
    }

    fn resolve_associated_type_with_params(
        &mut self,
        inner_ty: Ty,
        assoc_ty: Option<TypeAliasId>,
        // FIXME(GATs): these are args for the trait ref, args for assoc type itself should be
        // handled when we support them.
        params: &[GenericArg],
    ) -> Ty {
        match assoc_ty {
            Some(res_assoc_ty) => {
                let trait_ = match res_assoc_ty.lookup(self.db).container {
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

    fn resolve_variant(
        &mut self,
        node: ExprOrPatId,
        path: Option<&Path>,
        value_ns: bool,
    ) -> (Ty, Option<VariantId>) {
        let path = match path {
            Some(path) => path,
            None => return (self.err_ty(), None),
        };
        let mut ctx = TyLoweringContext::new(
            self.db,
            &self.resolver,
            &self.body.store,
            &self.diagnostics,
            InferenceTyDiagnosticSource::Body,
            self.generic_def,
            LifetimeElisionKind::Infer,
        );
        let mut path_ctx = ctx.at_path(path, node);
        let (resolution, unresolved) = if value_ns {
            let Some(res) = path_ctx.resolve_path_in_value_ns(HygieneId::ROOT) else {
                return (self.err_ty(), None);
            };
            match res {
                ResolveValueResult::ValueNs(value, _) => match value {
                    ValueNs::EnumVariantId(var) => {
                        let substs = path_ctx.substs_from_path(var.into(), true, false);
                        drop(ctx);
                        let ty = self.db.ty(var.lookup(self.db).parent.into());
                        let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                        return (ty, Some(var.into()));
                    }
                    ValueNs::StructId(strukt) => {
                        let substs = path_ctx.substs_from_path(strukt.into(), true, false);
                        drop(ctx);
                        let ty = self.db.ty(strukt.into());
                        let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                        return (ty, Some(strukt.into()));
                    }
                    ValueNs::ImplSelf(impl_id) => (TypeNs::SelfType(impl_id), None),
                    _ => {
                        drop(ctx);
                        return (self.err_ty(), None);
                    }
                },
                ResolveValueResult::Partial(typens, unresolved, _) => (typens, Some(unresolved)),
            }
        } else {
            match path_ctx.resolve_path_in_type_ns() {
                Some((it, idx)) => (it, idx),
                None => return (self.err_ty(), None),
            }
        };
        return match resolution {
            TypeNs::AdtId(AdtId::StructId(strukt)) => {
                let substs = path_ctx.substs_from_path(strukt.into(), true, false);
                drop(ctx);
                let ty = self.db.ty(strukt.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                forbid_unresolved_segments((ty, Some(strukt.into())), unresolved)
            }
            TypeNs::AdtId(AdtId::UnionId(u)) => {
                let substs = path_ctx.substs_from_path(u.into(), true, false);
                drop(ctx);
                let ty = self.db.ty(u.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                forbid_unresolved_segments((ty, Some(u.into())), unresolved)
            }
            TypeNs::EnumVariantId(var) => {
                let substs = path_ctx.substs_from_path(var.into(), true, false);
                drop(ctx);
                let ty = self.db.ty(var.lookup(self.db).parent.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));
                forbid_unresolved_segments((ty, Some(var.into())), unresolved)
            }
            TypeNs::SelfType(impl_id) => {
                let generics = crate::generics::generics(self.db, impl_id.into());
                let substs = generics.placeholder_subst(self.db);
                let mut ty = self.db.impl_self_ty(impl_id).substitute(Interner, &substs);

                let Some(remaining_idx) = unresolved else {
                    drop(ctx);
                    let Some(mod_path) = path.mod_path() else {
                        never!("resolver should always resolve lang item paths");
                        return (self.err_ty(), None);
                    };
                    return self.resolve_variant_on_alias(ty, None, mod_path);
                };

                let mut remaining_segments = path.segments().skip(remaining_idx);

                if remaining_segments.len() >= 2 {
                    path_ctx.ignore_last_segment();
                }

                // We need to try resolving unresolved segments one by one because each may resolve
                // to a projection, which `TyLoweringContext` cannot handle on its own.
                let mut tried_resolving_once = false;
                while let Some(current_segment) = remaining_segments.first() {
                    // If we can resolve to an enum variant, it takes priority over associated type
                    // of the same name.
                    if let Some((AdtId::EnumId(id), _)) = ty.as_adt() {
                        let enum_data = id.enum_variants(self.db);
                        if let Some(variant) = enum_data.variant(current_segment.name) {
                            return if remaining_segments.len() == 1 {
                                (ty, Some(variant.into()))
                            } else {
                                // We still have unresolved paths, but enum variants never have
                                // associated types!
                                // FIXME: Report an error.
                                (self.err_ty(), None)
                            };
                        }
                    }

                    if tried_resolving_once {
                        // FIXME: with `inherent_associated_types` this is allowed, but our `lower_partly_resolved_path()`
                        // will need to be updated to err at the correct segment.
                        break;
                    }

                    // `lower_partly_resolved_path()` returns `None` as type namespace unless
                    // `remaining_segments` is empty, which is never the case here. We don't know
                    // which namespace the new `ty` is in until normalized anyway.
                    (ty, _) = path_ctx.lower_partly_resolved_path(resolution, true);
                    tried_resolving_once = true;

                    ty = self.table.insert_type_vars(ty);
                    ty = self.table.normalize_associated_types_in(ty);
                    ty = self.table.resolve_ty_shallow(&ty);
                    if ty.is_unknown() {
                        return (self.err_ty(), None);
                    }

                    remaining_segments = remaining_segments.skip(1);
                }
                drop(ctx);

                let variant = ty.as_adt().and_then(|(id, _)| match id {
                    AdtId::StructId(s) => Some(VariantId::StructId(s)),
                    AdtId::UnionId(u) => Some(VariantId::UnionId(u)),
                    AdtId::EnumId(_) => {
                        // FIXME Error E0071, expected struct, variant or union type, found enum `Foo`
                        None
                    }
                });
                (ty, variant)
            }
            TypeNs::TypeAliasId(it) => {
                let Some(mod_path) = path.mod_path() else {
                    never!("resolver should always resolve lang item paths");
                    return (self.err_ty(), None);
                };
                let substs = path_ctx.substs_from_path_segment(it.into(), true, None, false);
                drop(ctx);
                let ty = self.db.ty(it.into());
                let ty = self.insert_type_vars(ty.substitute(Interner, &substs));

                self.resolve_variant_on_alias(ty, unresolved, mod_path)
            }
            TypeNs::AdtSelfType(_) => {
                // FIXME this could happen in array size expressions, once we're checking them
                (self.err_ty(), None)
            }
            TypeNs::GenericParam(_) => {
                // FIXME potentially resolve assoc type
                (self.err_ty(), None)
            }
            TypeNs::AdtId(AdtId::EnumId(_))
            | TypeNs::BuiltinType(_)
            | TypeNs::TraitId(_)
            | TypeNs::TraitAliasId(_)
            | TypeNs::ModuleId(_) => {
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
        path: &ModPath,
    ) -> (Ty, Option<VariantId>) {
        let remaining = unresolved.map(|it| path.segments()[it..].len()).filter(|it| it > &0);
        let ty = match ty.kind(Interner) {
            TyKind::Alias(AliasTy::Projection(proj_ty)) => {
                let ty = self.table.normalize_projection_ty(proj_ty.clone());
                self.table.resolve_ty_shallow(&ty)
            }
            _ => ty,
        };
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
                let segment = path.segments().last().unwrap();
                // this could be an enum variant or associated type
                if let Some((AdtId::EnumId(enum_id), _)) = ty.as_adt() {
                    let enum_data = enum_id.enum_variants(self.db);
                    if let Some(variant) = enum_data.variant(segment) {
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

    fn resolve_lang_item(&self, item: LangItem) -> Option<LangItemTarget> {
        let krate = self.resolver.krate();
        lang_item(self.db, krate, item)
    }

    fn resolve_output_on(&self, trait_: TraitId) -> Option<TypeAliasId> {
        trait_.trait_items(self.db).associated_type_by_name(&Name::new_symbol_root(sym::Output))
    }

    fn resolve_lang_trait(&self, lang: LangItem) -> Option<TraitId> {
        self.resolve_lang_item(lang)?.as_trait()
    }

    fn resolve_ops_neg_output(&self) -> Option<TypeAliasId> {
        self.resolve_output_on(self.resolve_lang_trait(LangItem::Neg)?)
    }

    fn resolve_ops_not_output(&self) -> Option<TypeAliasId> {
        self.resolve_output_on(self.resolve_lang_trait(LangItem::Not)?)
    }

    fn resolve_future_future_output(&self) -> Option<TypeAliasId> {
        let ItemContainerId::TraitId(trait_) = self
            .resolve_lang_item(LangItem::IntoFutureIntoFuture)?
            .as_function()?
            .lookup(self.db)
            .container
        else {
            return None;
        };
        self.resolve_output_on(trait_)
    }

    fn resolve_boxed_box(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::OwnedBox)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_full(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::RangeFull)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::Range)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_inclusive(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::RangeInclusiveStruct)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_from(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::RangeFrom)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_to(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::RangeTo)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_to_inclusive(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::RangeToInclusive)?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_ops_index_output(&self) -> Option<TypeAliasId> {
        self.resolve_output_on(self.resolve_lang_trait(LangItem::Index)?)
    }

    fn resolve_va_list(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item(LangItem::VaList)?.as_struct()?;
        Some(struct_.into())
    }

    fn get_traits_in_scope(&self) -> Either<FxHashSet<TraitId>, &FxHashSet<TraitId>> {
        let mut b_traits = self.resolver.traits_in_scope_from_block_scopes().peekable();
        if b_traits.peek().is_some() {
            Either::Left(self.traits_in_scope.iter().copied().chain(b_traits).collect())
        } else {
            Either::Right(&self.traits_in_scope)
        }
    }
}

/// When inferring an expression, we propagate downward whatever type hint we
/// are able in the form of an `Expectation`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Expectation {
    None,
    HasType(Ty),
    #[allow(dead_code)]
    Castable(Ty),
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

    /// The following explanation is copied straight from rustc:
    /// Provides an expectation for an rvalue expression given an *optional*
    /// hint, which is not required for type safety (the resulting type might
    /// be checked higher up, as is the case with `&expr` and `box expr`), but
    /// is useful in determining the concrete type.
    ///
    /// The primary use case is where the expected type is a fat pointer,
    /// like `&[isize]`. For example, consider the following statement:
    ///
    ///     let it: &[isize] = &[1, 2, 3];
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
    fn rvalue_hint(ctx: &mut InferenceContext<'_>, ty: Ty) -> Self {
        match ctx.struct_tail_without_normalization(ty.clone()).kind(Interner) {
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
            Expectation::Castable(t) => Expectation::Castable(table.resolve_ty_shallow(t)),
            Expectation::RValueLikeUnsized(t) => {
                Expectation::RValueLikeUnsized(table.resolve_ty_shallow(t))
            }
        }
    }

    fn to_option(&self, table: &mut unify::InferenceTable<'_>) -> Option<Ty> {
        match self.resolve(table) {
            Expectation::None => None,
            Expectation::HasType(t)
            | Expectation::Castable(t)
            | Expectation::RValueLikeUnsized(t) => Some(t),
        }
    }

    fn only_has_type(&self, table: &mut unify::InferenceTable<'_>) -> Option<Ty> {
        match self {
            Expectation::HasType(t) => Some(table.resolve_ty_shallow(t)),
            Expectation::Castable(_) | Expectation::RValueLikeUnsized(_) | Expectation::None => {
                None
            }
        }
    }

    fn coercion_target_type(&self, table: &mut unify::InferenceTable<'_>) -> Ty {
        self.only_has_type(table).unwrap_or_else(|| table.new_type_var())
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
                if ety.is_ty_var() { Expectation::None } else { Expectation::HasType(ety) }
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

/// A zipper that checks for unequal occurrences of `{unknown}` and unresolved projections
/// in the two types. Used to filter out mismatch diagnostics that only differ in
/// `{unknown}` and unresolved projections. These mismatches are usually not helpful.
/// As the cause is usually an underlying name resolution problem
struct UnknownMismatch<'db>(&'db dyn HirDatabase);
impl chalk_ir::zip::Zipper<Interner> for UnknownMismatch<'_> {
    fn zip_tys(&mut self, variance: Variance, a: &Ty, b: &Ty) -> chalk_ir::Fallible<()> {
        let zip_substs = |this: &mut Self,
                          variances,
                          sub_a: &Substitution,
                          sub_b: &Substitution| {
            this.zip_substs(variance, variances, sub_a.as_slice(Interner), sub_b.as_slice(Interner))
        };
        match (a.kind(Interner), b.kind(Interner)) {
            (TyKind::Adt(id_a, sub_a), TyKind::Adt(id_b, sub_b)) if id_a == id_b => zip_substs(
                self,
                Some(self.unification_database().adt_variance(*id_a)),
                sub_a,
                sub_b,
            )?,
            (
                TyKind::AssociatedType(assoc_ty_a, sub_a),
                TyKind::AssociatedType(assoc_ty_b, sub_b),
            ) if assoc_ty_a == assoc_ty_b => zip_substs(self, None, sub_a, sub_b)?,
            (TyKind::Tuple(arity_a, sub_a), TyKind::Tuple(arity_b, sub_b))
                if arity_a == arity_b =>
            {
                zip_substs(self, None, sub_a, sub_b)?
            }
            (TyKind::OpaqueType(opaque_ty_a, sub_a), TyKind::OpaqueType(opaque_ty_b, sub_b))
                if opaque_ty_a == opaque_ty_b =>
            {
                zip_substs(self, None, sub_a, sub_b)?
            }
            (TyKind::Slice(ty_a), TyKind::Slice(ty_b)) => self.zip_tys(variance, ty_a, ty_b)?,
            (TyKind::FnDef(fn_def_a, sub_a), TyKind::FnDef(fn_def_b, sub_b))
                if fn_def_a == fn_def_b =>
            {
                zip_substs(
                    self,
                    Some(self.unification_database().fn_def_variance(*fn_def_a)),
                    sub_a,
                    sub_b,
                )?
            }
            (TyKind::Ref(mutability_a, _, ty_a), TyKind::Ref(mutability_b, _, ty_b))
                if mutability_a == mutability_b =>
            {
                self.zip_tys(variance, ty_a, ty_b)?
            }
            (TyKind::Raw(mutability_a, ty_a), TyKind::Raw(mutability_b, ty_b))
                if mutability_a == mutability_b =>
            {
                self.zip_tys(variance, ty_a, ty_b)?
            }
            (TyKind::Array(ty_a, const_a), TyKind::Array(ty_b, const_b)) if const_a == const_b => {
                self.zip_tys(variance, ty_a, ty_b)?
            }
            (TyKind::Closure(id_a, sub_a), TyKind::Closure(id_b, sub_b)) if id_a == id_b => {
                zip_substs(self, None, sub_a, sub_b)?
            }
            (TyKind::Coroutine(coroutine_a, sub_a), TyKind::Coroutine(coroutine_b, sub_b))
                if coroutine_a == coroutine_b =>
            {
                zip_substs(self, None, sub_a, sub_b)?
            }
            (
                TyKind::CoroutineWitness(coroutine_a, sub_a),
                TyKind::CoroutineWitness(coroutine_b, sub_b),
            ) if coroutine_a == coroutine_b => zip_substs(self, None, sub_a, sub_b)?,
            (TyKind::Function(fn_ptr_a), TyKind::Function(fn_ptr_b))
                if fn_ptr_a.sig == fn_ptr_b.sig && fn_ptr_a.num_binders == fn_ptr_b.num_binders =>
            {
                zip_substs(self, None, &fn_ptr_a.substitution.0, &fn_ptr_b.substitution.0)?
            }
            (TyKind::Error, TyKind::Error) => (),
            (TyKind::Error, _)
            | (_, TyKind::Error)
            | (TyKind::Alias(AliasTy::Projection(_)) | TyKind::AssociatedType(_, _), _)
            | (_, TyKind::Alias(AliasTy::Projection(_)) | TyKind::AssociatedType(_, _)) => {
                return Err(chalk_ir::NoSolution);
            }
            _ => (),
        }

        Ok(())
    }

    fn zip_lifetimes(&mut self, _: Variance, _: &Lifetime, _: &Lifetime) -> chalk_ir::Fallible<()> {
        Ok(())
    }

    fn zip_consts(&mut self, _: Variance, _: &Const, _: &Const) -> chalk_ir::Fallible<()> {
        Ok(())
    }

    fn zip_binders<T>(
        &mut self,
        variance: Variance,
        a: &Binders<T>,
        b: &Binders<T>,
    ) -> chalk_ir::Fallible<()>
    where
        T: Clone
            + HasInterner<Interner = Interner>
            + chalk_ir::zip::Zip<Interner>
            + TypeFoldable<Interner>,
    {
        chalk_ir::zip::Zip::zip_with(self, variance, a.skip_binders(), b.skip_binders())
    }

    fn interner(&self) -> Interner {
        Interner
    }

    fn unification_database(&self) -> &dyn chalk_ir::UnificationDatabase<Interner> {
        &self.0
    }
}
