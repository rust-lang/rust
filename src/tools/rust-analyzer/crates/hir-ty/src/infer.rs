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

mod autoderef;
pub(crate) mod cast;
pub(crate) mod closure;
mod coerce;
pub(crate) mod diagnostics;
mod expr;
mod fallback;
mod mutability;
mod opaques;
mod pat;
mod path;
pub(crate) mod unify;

use std::{cell::OnceCell, convert::identity, iter, ops::Index};

use base_db::Crate;
use either::Either;
use hir_def::{
    AdtId, AssocItemId, ConstId, DefWithBodyId, FieldId, FunctionId, GenericDefId, GenericParamId,
    ItemContainerId, LocalFieldId, Lookup, TraitId, TupleFieldId, TupleId, TypeAliasId, VariantId,
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
use la_arena::ArenaMap;
use rustc_ast_ir::Mutability;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    AliasTyKind, TypeFoldable,
    inherent::{AdtDef, IntoKind, Region as _, SliceLike, Ty as _},
};
use stdx::never;
use triomphe::Arc;

use crate::{
    ImplTraitId, IncorrectGenericsLenKind, PathLoweringDiagnostic, TargetFeatures,
    db::{HirDatabase, InternedClosureId, InternedOpaqueTyId},
    infer::{
        coerce::{CoerceMany, DynamicCoerceMany},
        diagnostics::{Diagnostics, InferenceTyLoweringContext as TyLoweringContext},
        expr::ExprIsRead,
    },
    lower::{
        ImplTraitIdx, ImplTraitLoweringMode, LifetimeElisionKind, diagnostics::TyLoweringDiagnostic,
    },
    mir::MirSpan,
    next_solver::{
        AliasTy, Const, DbInterner, ErrorGuaranteed, GenericArg, GenericArgs, Region, Ty, TyKind,
        Tys, abi::Safety, infer::traits::ObligationCause,
    },
    traits::FnTrait,
    utils::TargetFeatureIsSafeInTarget,
};

// This lint has a false positive here. See the link below for details.
//
// https://github.com/rust-lang/rust/issues/57411
#[allow(unreachable_pub)]
pub use coerce::could_coerce;
#[allow(unreachable_pub)]
pub use unify::{could_unify, could_unify_deeply};

use cast::{CastCheck, CastError};
pub(crate) use closure::analysis::{CaptureKind, CapturedItem, CapturedItemWithoutTy};

/// The entry point of type inference.
pub(crate) fn infer_query(db: &dyn HirDatabase, def: DefWithBodyId) -> Arc<InferenceResult<'_>> {
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
            ctx.return_ty = match db.enum_signature(v.lookup(db).parent).variant_body_type() {
                hir_def::layout::IntegerType::Pointer(signed) => match signed {
                    true => ctx.types.isize,
                    false => ctx.types.usize,
                },
                hir_def::layout::IntegerType::Fixed(size, signed) => match signed {
                    true => match size {
                        Integer::I8 => ctx.types.i8,
                        Integer::I16 => ctx.types.i16,
                        Integer::I32 => ctx.types.i32,
                        Integer::I64 => ctx.types.i64,
                        Integer::I128 => ctx.types.i128,
                    },
                    false => match size {
                        Integer::I8 => ctx.types.u8,
                        Integer::I16 => ctx.types.u16,
                        Integer::I32 => ctx.types.u32,
                        Integer::I64 => ctx.types.u64,
                        Integer::I128 => ctx.types.u128,
                    },
                },
            };
        }
    }

    ctx.infer_body();

    ctx.infer_mut_body();

    ctx.handle_opaque_type_uses();

    ctx.type_inference_fallback();

    // Comment from rustc:
    // Even though coercion casts provide type hints, we check casts after fallback for
    // backwards compatibility. This makes fallback a stronger type hint than a cast coercion.
    let cast_checks = std::mem::take(&mut ctx.deferred_cast_checks);
    for mut cast in cast_checks.into_iter() {
        if let Err(diag) = cast.check(&mut ctx) {
            ctx.diagnostics.push(diag);
        }
    }

    ctx.table.select_obligations_where_possible();

    ctx.infer_closures();

    ctx.table.select_obligations_where_possible();

    ctx.handle_opaque_type_uses();

    Arc::new(ctx.resolve_all())
}

pub(crate) fn infer_cycle_result(
    db: &dyn HirDatabase,
    _: DefWithBodyId,
) -> Arc<InferenceResult<'_>> {
    Arc::new(InferenceResult {
        has_errors: true,
        ..InferenceResult::new(Ty::new_error(DbInterner::new_with(db, None, None), ErrorGuaranteed))
    })
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InferenceTyDiagnosticSource {
    /// Diagnostics that come from types in the body.
    Body,
    /// Diagnostics that come from types in fn parameters/return type, or static & const types.
    Signature,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum InferenceDiagnostic<'db> {
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
        receiver: Ty<'db>,
        name: Name,
        method_with_same_name_exists: bool,
    },
    UnresolvedMethodCall {
        expr: ExprId,
        receiver: Ty<'db>,
        name: Name,
        /// Contains the type the field resolves to
        field_with_same_name: Option<Ty<'db>>,
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
        found: Ty<'db>,
    },
    TypedHole {
        expr: ExprId,
        expected: Ty<'db>,
    },
    CastToUnsized {
        expr: ExprId,
        cast_ty: Ty<'db>,
    },
    InvalidCast {
        expr: ExprId,
        error: CastError,
        expr_ty: Ty<'db>,
        cast_ty: Ty<'db>,
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
pub struct TypeMismatch<'db> {
    pub expected: Ty<'db>,
    pub actual: Ty<'db>,
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
pub struct Adjustment<'db> {
    pub kind: Adjust<'db>,
    pub target: Ty<'db>,
}

impl<'db> Adjustment<'db> {
    pub fn borrow(interner: DbInterner<'db>, m: Mutability, ty: Ty<'db>, lt: Region<'db>) -> Self {
        let ty = Ty::new_ref(interner, lt, ty, m);
        Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(lt, m)), target: ty }
    }
}

/// At least for initial deployment, we want to limit two-phase borrows to
/// only a few specific cases. Right now, those are mostly "things that desugar"
/// into method calls:
/// - using `x.some_method()` syntax, where some_method takes `&mut self`,
/// - using `Foo::some_method(&mut x, ...)` syntax,
/// - binary assignment operators (`+=`, `-=`, `*=`, etc.).
///
/// Anything else should be rejected until generalized two-phase borrow support
/// is implemented. Right now, dataflow can't handle the general case where there
/// is more than one use of a mutable borrow, and we don't want to accept too much
/// new code via two-phase borrows, so we try to limit where we create two-phase
/// capable mutable borrows.
/// See #49434 for tracking.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum AllowTwoPhase {
    // FIXME: We should use this when appropriate.
    Yes,
    No,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Adjust<'db> {
    /// Go from ! to any type.
    NeverToAny,
    /// Dereference once, producing a place.
    Deref(Option<OverloadedDeref>),
    /// Take the address and produce either a `&` or `*` pointer.
    Borrow(AutoBorrow<'db>),
    Pointer(PointerCast),
}

/// An overloaded autoderef step, representing a `Deref(Mut)::deref(_mut)`
/// call, with the signature `&'a T -> &'a U` or `&'a mut T -> &'a mut U`.
/// The target type is `U` in both cases, with the region and mutability
/// being those shared by both the receiver and the returned reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverloadedDeref(pub Option<Mutability>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AutoBorrow<'db> {
    /// Converts from T to &T.
    Ref(Region<'db>, Mutability),
    /// Converts from T to *T.
    RawPtr(Mutability),
}

impl<'db> AutoBorrow<'db> {
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InferenceResult<'db> {
    /// For each method call expr, records the function it resolves to.
    method_resolutions: FxHashMap<ExprId, (FunctionId, GenericArgs<'db>)>,
    /// For each field access expr, records the field it resolves to.
    field_resolutions: FxHashMap<ExprId, Either<FieldId, TupleFieldId>>,
    /// For each struct literal or pattern, records the variant it resolves to.
    variant_resolutions: FxHashMap<ExprOrPatId, VariantId>,
    /// For each associated item record what it resolves to
    assoc_resolutions: FxHashMap<ExprOrPatId, (AssocItemId, GenericArgs<'db>)>,
    /// Whenever a tuple field expression access a tuple field, we allocate a tuple id in
    /// [`InferenceContext`] and store the tuples substitution there. This map is the reverse of
    /// that which allows us to resolve a [`TupleFieldId`]s type.
    tuple_field_access_types: FxHashMap<TupleId, Tys<'db>>,
    /// During inference this field is empty and [`InferenceContext::diagnostics`] is filled instead.
    diagnostics: Vec<InferenceDiagnostic<'db>>,
    pub(crate) type_of_expr: ArenaMap<ExprId, Ty<'db>>,
    /// For each pattern record the type it resolves to.
    ///
    /// **Note**: When a pattern type is resolved it may still contain
    /// unresolved or missing subpatterns or subpatterns of mismatched types.
    pub(crate) type_of_pat: ArenaMap<PatId, Ty<'db>>,
    pub(crate) type_of_binding: ArenaMap<BindingId, Ty<'db>>,
    pub(crate) type_of_opaque: FxHashMap<InternedOpaqueTyId, Ty<'db>>,
    type_mismatches: FxHashMap<ExprOrPatId, TypeMismatch<'db>>,
    /// Whether there are any type-mismatching errors in the result.
    // FIXME: This isn't as useful as initially thought due to us falling back placeholders to
    // `TyKind::Error`.
    // Which will then mark this field.
    pub(crate) has_errors: bool,
    /// Interned `Error` type to return references to.
    // FIXME: Remove this.
    error_ty: Ty<'db>,
    /// Stores the types which were implicitly dereferenced in pattern binding modes.
    pub(crate) pat_adjustments: FxHashMap<PatId, Vec<Ty<'db>>>,
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
    pub(crate) binding_modes: ArenaMap<PatId, BindingMode>,
    pub(crate) expr_adjustments: FxHashMap<ExprId, Box<[Adjustment<'db>]>>,
    pub(crate) closure_info: FxHashMap<InternedClosureId, (Vec<CapturedItem<'db>>, FnTrait)>,
    // FIXME: remove this field
    pub mutated_bindings_in_closure: FxHashSet<BindingId>,
    pub(crate) coercion_casts: FxHashSet<ExprId>,
}

impl<'db> InferenceResult<'db> {
    fn new(error_ty: Ty<'db>) -> Self {
        Self {
            method_resolutions: Default::default(),
            field_resolutions: Default::default(),
            variant_resolutions: Default::default(),
            assoc_resolutions: Default::default(),
            tuple_field_access_types: Default::default(),
            diagnostics: Default::default(),
            type_of_expr: Default::default(),
            type_of_pat: Default::default(),
            type_of_binding: Default::default(),
            type_of_opaque: Default::default(),
            type_mismatches: Default::default(),
            has_errors: Default::default(),
            error_ty,
            pat_adjustments: Default::default(),
            binding_modes: Default::default(),
            expr_adjustments: Default::default(),
            closure_info: Default::default(),
            mutated_bindings_in_closure: Default::default(),
            coercion_casts: Default::default(),
        }
    }

    pub fn method_resolution(&self, expr: ExprId) -> Option<(FunctionId, GenericArgs<'db>)> {
        self.method_resolutions.get(&expr).copied()
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
    pub fn assoc_resolutions_for_expr(
        &self,
        id: ExprId,
    ) -> Option<(AssocItemId, GenericArgs<'db>)> {
        self.assoc_resolutions.get(&id.into()).copied()
    }
    pub fn assoc_resolutions_for_pat(&self, id: PatId) -> Option<(AssocItemId, GenericArgs<'db>)> {
        self.assoc_resolutions.get(&id.into()).copied()
    }
    pub fn assoc_resolutions_for_expr_or_pat(
        &self,
        id: ExprOrPatId,
    ) -> Option<(AssocItemId, GenericArgs<'db>)> {
        match id {
            ExprOrPatId::ExprId(id) => self.assoc_resolutions_for_expr(id),
            ExprOrPatId::PatId(id) => self.assoc_resolutions_for_pat(id),
        }
    }
    pub fn type_mismatch_for_expr(&self, expr: ExprId) -> Option<&TypeMismatch<'db>> {
        self.type_mismatches.get(&expr.into())
    }
    pub fn type_mismatch_for_pat(&self, pat: PatId) -> Option<&TypeMismatch<'db>> {
        self.type_mismatches.get(&pat.into())
    }
    pub fn type_mismatches(&self) -> impl Iterator<Item = (ExprOrPatId, &TypeMismatch<'db>)> {
        self.type_mismatches.iter().map(|(expr_or_pat, mismatch)| (*expr_or_pat, mismatch))
    }
    pub fn expr_type_mismatches(&self) -> impl Iterator<Item = (ExprId, &TypeMismatch<'db>)> {
        self.type_mismatches.iter().filter_map(|(expr_or_pat, mismatch)| match *expr_or_pat {
            ExprOrPatId::ExprId(expr) => Some((expr, mismatch)),
            _ => None,
        })
    }
    pub fn closure_info(&self, closure: InternedClosureId) -> &(Vec<CapturedItem<'db>>, FnTrait) {
        self.closure_info.get(&closure).unwrap()
    }
    pub fn type_of_expr_or_pat(&self, id: ExprOrPatId) -> Option<Ty<'db>> {
        match id {
            ExprOrPatId::ExprId(id) => self.type_of_expr.get(id).copied(),
            ExprOrPatId::PatId(id) => self.type_of_pat.get(id).copied(),
        }
    }
    pub fn type_of_expr_with_adjust(&self, id: ExprId) -> Option<Ty<'db>> {
        match self.expr_adjustments.get(&id).and_then(|adjustments| {
            adjustments
                .iter()
                .filter(|adj| {
                    // https://github.com/rust-lang/rust/blob/67819923ac8ea353aaa775303f4c3aacbf41d010/compiler/rustc_mir_build/src/thir/cx/expr.rs#L140
                    !matches!(
                        adj,
                        Adjustment {
                            kind: Adjust::NeverToAny,
                            target,
                        } if target.is_never()
                    )
                })
                .next_back()
        }) {
            Some(adjustment) => Some(adjustment.target),
            None => self.type_of_expr.get(id).copied(),
        }
    }
    pub fn type_of_pat_with_adjust(&self, id: PatId) -> Option<Ty<'db>> {
        match self.pat_adjustments.get(&id).and_then(|adjustments| adjustments.last()) {
            Some(adjusted) => Some(*adjusted),
            None => self.type_of_pat.get(id).copied(),
        }
    }
    pub fn is_erroneous(&self) -> bool {
        self.has_errors && self.type_of_expr.iter().count() == 0
    }

    pub fn diagnostics(&self) -> &[InferenceDiagnostic<'db>] {
        &self.diagnostics
    }

    pub fn tuple_field_access_type(&self, id: TupleId) -> Tys<'db> {
        self.tuple_field_access_types[&id]
    }

    pub fn pat_adjustment(&self, id: PatId) -> Option<&[Ty<'db>]> {
        self.pat_adjustments.get(&id).map(|it| &**it)
    }

    pub fn expr_adjustment(&self, id: ExprId) -> Option<&[Adjustment<'db>]> {
        self.expr_adjustments.get(&id).map(|it| &**it)
    }

    pub fn binding_mode(&self, id: PatId) -> Option<BindingMode> {
        self.binding_modes.get(id).copied()
    }

    // This method is consumed by external tools to run rust-analyzer as a library. Don't remove, please.
    pub fn expression_types(&self) -> impl Iterator<Item = (ExprId, Ty<'db>)> {
        self.type_of_expr.iter().map(|(k, v)| (k, *v))
    }

    // This method is consumed by external tools to run rust-analyzer as a library. Don't remove, please.
    pub fn pattern_types(&self) -> impl Iterator<Item = (PatId, Ty<'db>)> {
        self.type_of_pat.iter().map(|(k, v)| (k, *v))
    }

    // This method is consumed by external tools to run rust-analyzer as a library. Don't remove, please.
    pub fn binding_types(&self) -> impl Iterator<Item = (BindingId, Ty<'db>)> {
        self.type_of_binding.iter().map(|(k, v)| (k, *v))
    }

    // This method is consumed by external tools to run rust-analyzer as a library. Don't remove, please.
    pub fn return_position_impl_trait_types(
        &self,
        db: &'db dyn HirDatabase,
    ) -> impl Iterator<Item = (ImplTraitIdx<'db>, Ty<'db>)> {
        self.type_of_opaque.iter().filter_map(move |(&id, &ty)| {
            let ImplTraitId::ReturnTypeImplTrait(_, rpit_idx) = id.loc(db) else {
                return None;
            };
            Some((rpit_idx, ty))
        })
    }
}

impl<'db> Index<ExprId> for InferenceResult<'db> {
    type Output = Ty<'db>;

    fn index(&self, expr: ExprId) -> &Ty<'db> {
        self.type_of_expr.get(expr).unwrap_or(&self.error_ty)
    }
}

impl<'db> Index<PatId> for InferenceResult<'db> {
    type Output = Ty<'db>;

    fn index(&self, pat: PatId) -> &Ty<'db> {
        self.type_of_pat.get(pat).unwrap_or(&self.error_ty)
    }
}

impl<'db> Index<ExprOrPatId> for InferenceResult<'db> {
    type Output = Ty<'db>;

    fn index(&self, id: ExprOrPatId) -> &Ty<'db> {
        match id {
            ExprOrPatId::ExprId(id) => &self[id],
            ExprOrPatId::PatId(id) => &self[id],
        }
    }
}

impl<'db> Index<BindingId> for InferenceResult<'db> {
    type Output = Ty<'db>;

    fn index(&self, b: BindingId) -> &Ty<'db> {
        self.type_of_binding.get(b).unwrap_or(&self.error_ty)
    }
}

#[derive(Debug, Clone)]
struct InternedStandardTypes<'db> {
    unit: Ty<'db>,
    never: Ty<'db>,
    char: Ty<'db>,
    bool: Ty<'db>,
    i8: Ty<'db>,
    i16: Ty<'db>,
    i32: Ty<'db>,
    i64: Ty<'db>,
    i128: Ty<'db>,
    isize: Ty<'db>,
    u8: Ty<'db>,
    u16: Ty<'db>,
    u32: Ty<'db>,
    u64: Ty<'db>,
    u128: Ty<'db>,
    usize: Ty<'db>,
    f16: Ty<'db>,
    f32: Ty<'db>,
    f64: Ty<'db>,
    f128: Ty<'db>,
    static_str_ref: Ty<'db>,
    error: Ty<'db>,

    re_static: Region<'db>,
    re_error: Region<'db>,
    re_erased: Region<'db>,

    empty_args: GenericArgs<'db>,
    empty_tys: Tys<'db>,
}

impl<'db> InternedStandardTypes<'db> {
    fn new(interner: DbInterner<'db>) -> Self {
        let str = Ty::new(interner, rustc_type_ir::TyKind::Str);
        let re_static = Region::new_static(interner);
        Self {
            unit: Ty::new_unit(interner),
            never: Ty::new(interner, TyKind::Never),
            char: Ty::new(interner, TyKind::Char),
            bool: Ty::new(interner, TyKind::Bool),
            i8: Ty::new_int(interner, rustc_type_ir::IntTy::I8),
            i16: Ty::new_int(interner, rustc_type_ir::IntTy::I16),
            i32: Ty::new_int(interner, rustc_type_ir::IntTy::I32),
            i64: Ty::new_int(interner, rustc_type_ir::IntTy::I64),
            i128: Ty::new_int(interner, rustc_type_ir::IntTy::I128),
            isize: Ty::new_int(interner, rustc_type_ir::IntTy::Isize),
            u8: Ty::new_uint(interner, rustc_type_ir::UintTy::U8),
            u16: Ty::new_uint(interner, rustc_type_ir::UintTy::U16),
            u32: Ty::new_uint(interner, rustc_type_ir::UintTy::U32),
            u64: Ty::new_uint(interner, rustc_type_ir::UintTy::U64),
            u128: Ty::new_uint(interner, rustc_type_ir::UintTy::U128),
            usize: Ty::new_uint(interner, rustc_type_ir::UintTy::Usize),
            f16: Ty::new_float(interner, rustc_type_ir::FloatTy::F16),
            f32: Ty::new_float(interner, rustc_type_ir::FloatTy::F32),
            f64: Ty::new_float(interner, rustc_type_ir::FloatTy::F64),
            f128: Ty::new_float(interner, rustc_type_ir::FloatTy::F128),
            static_str_ref: Ty::new_ref(interner, re_static, str, Mutability::Not),
            error: Ty::new_error(interner, ErrorGuaranteed),

            re_static,
            re_error: Region::error(interner),
            re_erased: Region::new_erased(interner),

            empty_args: GenericArgs::new_from_iter(interner, []),
            empty_tys: Tys::new_from_iter(interner, []),
        }
    }
}

/// The inference context contains all information needed during type inference.
#[derive(Clone, Debug)]
pub(crate) struct InferenceContext<'body, 'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) owner: DefWithBodyId,
    pub(crate) body: &'body Body,
    /// Generally you should not resolve things via this resolver. Instead create a TyLoweringContext
    /// and resolve the path via its methods. This will ensure proper error reporting.
    pub(crate) resolver: Resolver<'db>,
    target_features: OnceCell<(TargetFeatures, TargetFeatureIsSafeInTarget)>,
    pub(crate) generic_def: GenericDefId,
    table: unify::InferenceTable<'db>,
    /// The traits in scope, disregarding block modules. This is used for caching purposes.
    traits_in_scope: FxHashSet<TraitId>,
    pub(crate) result: InferenceResult<'db>,
    tuple_field_accesses_rev:
        IndexSet<Tys<'db>, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>,
    /// The return type of the function being inferred, the closure or async block if we're
    /// currently within one.
    ///
    /// We might consider using a nested inference context for checking
    /// closures so we can swap all shared things out at once.
    return_ty: Ty<'db>,
    /// If `Some`, this stores coercion information for returned
    /// expressions. If `None`, this is in a context where return is
    /// inappropriate, such as a const expression.
    return_coercion: Option<DynamicCoerceMany<'db>>,
    /// The resume type and the yield type, respectively, of the coroutine being inferred.
    resume_yield_tys: Option<(Ty<'db>, Ty<'db>)>,
    diverges: Diverges,
    breakables: Vec<BreakableContext<'db>>,
    types: InternedStandardTypes<'db>,

    /// Whether we are inside the pattern of a destructuring assignment.
    inside_assignment: bool,

    deferred_cast_checks: Vec<CastCheck<'db>>,

    // fields related to closure capture
    current_captures: Vec<CapturedItemWithoutTy<'db>>,
    /// A stack that has an entry for each projection in the current capture.
    ///
    /// For example, in `a.b.c`, we capture the spans of `a`, `a.b`, and `a.b.c`.
    /// We do that because sometimes we truncate projections (when a closure captures
    /// both `a.b` and `a.b.c`), and we want to provide accurate spans in this case.
    current_capture_span_stack: Vec<MirSpan>,
    current_closure: Option<InternedClosureId>,
    /// Stores the list of closure ids that need to be analyzed before this closure. See the
    /// comment on `InferenceContext::sort_closures`
    closure_dependencies: FxHashMap<InternedClosureId, Vec<InternedClosureId>>,
    deferred_closures: FxHashMap<InternedClosureId, Vec<(Ty<'db>, Ty<'db>, Vec<Ty<'db>>, ExprId)>>,

    diagnostics: Diagnostics<'db>,
}

#[derive(Clone, Debug)]
struct BreakableContext<'db> {
    /// Whether this context contains at least one break expression.
    may_break: bool,
    /// The coercion target of the context.
    coerce: Option<DynamicCoerceMany<'db>>,
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

fn find_breakable<'a, 'db>(
    ctxs: &'a mut [BreakableContext<'db>],
    label: Option<LabelId>,
) -> Option<&'a mut BreakableContext<'db>> {
    let mut ctxs = ctxs
        .iter_mut()
        .rev()
        .take_while(|it| matches!(it.kind, BreakableKind::Block | BreakableKind::Loop));
    match label {
        Some(_) => ctxs.find(|ctx| ctx.label == label),
        None => ctxs.find(|ctx| matches!(ctx.kind, BreakableKind::Loop)),
    }
}

fn find_continuable<'a, 'db>(
    ctxs: &'a mut [BreakableContext<'db>],
    label: Option<LabelId>,
) -> Option<&'a mut BreakableContext<'db>> {
    match label {
        Some(_) => find_breakable(ctxs, label).filter(|it| matches!(it.kind, BreakableKind::Loop)),
        None => find_breakable(ctxs, label),
    }
}

impl<'body, 'db> InferenceContext<'body, 'db> {
    fn new(
        db: &'db dyn HirDatabase,
        owner: DefWithBodyId,
        body: &'body Body,
        resolver: Resolver<'db>,
    ) -> Self {
        let trait_env = db.trait_environment_for_body(owner);
        let table = unify::InferenceTable::new(db, trait_env, Some(owner));
        let types = InternedStandardTypes::new(table.interner());
        InferenceContext {
            result: InferenceResult::new(types.error),
            return_ty: types.error, // set in collect_* calls
            types,
            target_features: OnceCell::new(),
            table,
            tuple_field_accesses_rev: Default::default(),
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

    #[inline]
    fn krate(&self) -> Crate {
        self.resolver.krate()
    }

    fn target_features<'a>(
        db: &dyn HirDatabase,
        target_features: &'a OnceCell<(TargetFeatures, TargetFeatureIsSafeInTarget)>,
        owner: DefWithBodyId,
        krate: Crate,
    ) -> (&'a TargetFeatures, TargetFeatureIsSafeInTarget) {
        let (target_features, target_feature_is_safe) = target_features.get_or_init(|| {
            let target_features = match owner {
                DefWithBodyId::FunctionId(id) => TargetFeatures::from_attrs(&db.attrs(id.into())),
                _ => TargetFeatures::default(),
            };
            let target_feature_is_safe = match &krate.workspace_data(db).target {
                Ok(target) => crate::utils::target_feature_is_safe_in_target(target),
                Err(_) => TargetFeatureIsSafeInTarget::No,
            };
            (target_features, target_feature_is_safe)
        });
        (target_features, *target_feature_is_safe)
    }

    #[inline]
    pub(crate) fn set_tainted_by_errors(&mut self) {
        self.result.has_errors = true;
    }

    /// Clones `self` and calls `resolve_all()` on it.
    // FIXME: Remove this.
    pub(crate) fn fixme_resolve_all_clone(&self) -> InferenceResult<'db> {
        let mut ctx = self.clone();

        ctx.type_inference_fallback();

        // Comment from rustc:
        // Even though coercion casts provide type hints, we check casts after fallback for
        // backwards compatibility. This makes fallback a stronger type hint than a cast coercion.
        let cast_checks = std::mem::take(&mut ctx.deferred_cast_checks);
        for mut cast in cast_checks.into_iter() {
            if let Err(diag) = cast.check(&mut ctx) {
                ctx.diagnostics.push(diag);
            }
        }

        ctx.table.select_obligations_where_possible();

        ctx.resolve_all()
    }

    // FIXME: This function should be private in module. It is currently only used in the consteval, since we need
    // `InferenceResult` in the middle of inference. See the fixme comment in `consteval::eval_to_const`. If you
    // used this function for another workaround, mention it here. If you really need this function and believe that
    // there is no problem in it being `pub(crate)`, remove this comment.
    fn resolve_all(self) -> InferenceResult<'db> {
        let InferenceContext {
            mut table, mut result, tuple_field_accesses_rev, diagnostics, ..
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
            type_of_opaque,
            type_mismatches,
            has_errors,
            error_ty: _,
            pat_adjustments,
            binding_modes: _,
            expr_adjustments,
            // Types in `closure_info` have already been `resolve_completely()`'d during
            // `InferenceContext::infer_closures()` (in `HirPlace::ty()` specifically), so no need
            // to resolve them here.
            closure_info: _,
            mutated_bindings_in_closure: _,
            tuple_field_access_types: _,
            coercion_casts: _,
            diagnostics: _,
        } = &mut result;

        for ty in type_of_expr.values_mut() {
            *ty = table.resolve_completely(*ty);
            *has_errors = *has_errors || ty.references_non_lt_error();
        }
        type_of_expr.shrink_to_fit();
        for ty in type_of_pat.values_mut() {
            *ty = table.resolve_completely(*ty);
            *has_errors = *has_errors || ty.references_non_lt_error();
        }
        type_of_pat.shrink_to_fit();
        for ty in type_of_binding.values_mut() {
            *ty = table.resolve_completely(*ty);
            *has_errors = *has_errors || ty.references_non_lt_error();
        }
        type_of_binding.shrink_to_fit();
        type_of_opaque.shrink_to_fit();

        *has_errors |= !type_mismatches.is_empty();

        for mismatch in (*type_mismatches).values_mut() {
            mismatch.expected = table.resolve_completely(mismatch.expected);
            mismatch.actual = table.resolve_completely(mismatch.actual);
        }
        type_mismatches.shrink_to_fit();
        diagnostics.retain_mut(|diagnostic| {
            use InferenceDiagnostic::*;
            match diagnostic {
                ExpectedFunction { found: ty, .. }
                | UnresolvedField { receiver: ty, .. }
                | UnresolvedMethodCall { receiver: ty, .. } => {
                    *ty = table.resolve_completely(*ty);
                    // FIXME: Remove this when we are on par with rustc in terms of inference
                    if ty.references_non_lt_error() {
                        return false;
                    }

                    if let UnresolvedMethodCall { field_with_same_name, .. } = diagnostic
                        && let Some(ty) = field_with_same_name
                    {
                        *ty = table.resolve_completely(*ty);
                        if ty.references_non_lt_error() {
                            *field_with_same_name = None;
                        }
                    }
                }
                TypedHole { expected: ty, .. } => {
                    *ty = table.resolve_completely(*ty);
                }
                _ => (),
            }
            true
        });
        diagnostics.shrink_to_fit();
        for (_, subst) in method_resolutions.values_mut() {
            *subst = table.resolve_completely(*subst);
            *has_errors = *has_errors || subst.types().any(|ty| ty.references_non_lt_error());
        }
        method_resolutions.shrink_to_fit();
        for (_, subst) in assoc_resolutions.values_mut() {
            *subst = table.resolve_completely(*subst);
            *has_errors = *has_errors || subst.types().any(|ty| ty.references_non_lt_error());
        }
        assoc_resolutions.shrink_to_fit();
        for adjustment in expr_adjustments.values_mut().flatten() {
            adjustment.target = table.resolve_completely(adjustment.target);
            *has_errors = *has_errors || adjustment.target.references_non_lt_error();
        }
        expr_adjustments.shrink_to_fit();
        for adjustment in pat_adjustments.values_mut().flatten() {
            *adjustment = table.resolve_completely(*adjustment);
            *has_errors = *has_errors || adjustment.references_non_lt_error();
        }
        pat_adjustments.shrink_to_fit();
        result.tuple_field_access_types = tuple_field_accesses_rev
            .into_iter()
            .enumerate()
            .map(|(idx, subst)| (TupleId(idx as u32), table.resolve_completely(subst)))
            .inspect(|(_, subst)| {
                *has_errors = *has_errors || subst.iter().any(|ty| ty.references_non_lt_error());
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
            LifetimeElisionKind::for_const(self.interner(), id.loc(self.db).container),
        );

        self.return_ty = return_ty;
    }

    fn collect_static(&mut self, data: &StaticSignature) {
        let return_ty = self.make_ty(
            data.type_ref,
            &data.store,
            InferenceTyDiagnosticSource::Signature,
            LifetimeElisionKind::Elided(self.types.re_static),
        );

        self.return_ty = return_ty;
    }

    fn collect_fn(&mut self, func: FunctionId) {
        let data = self.db.function_signature(func);
        let mut param_tys = self.with_ty_lowering(
            &data.store,
            InferenceTyDiagnosticSource::Signature,
            LifetimeElisionKind::for_fn_params(&data),
            |ctx| data.params.iter().map(|&type_ref| ctx.lower_ty(type_ref)).collect::<Vec<_>>(),
        );

        // Check if function contains a va_list, if it does then we append it to the parameter types
        // that are collected from the function data
        if data.is_varargs() {
            let va_list_ty = match self.resolve_va_list() {
                Some(va_list) => Ty::new_adt(
                    self.interner(),
                    va_list,
                    GenericArgs::for_item_with_defaults(
                        self.interner(),
                        va_list.into(),
                        |_, id, _| self.table.next_var_for_param(id),
                    ),
                ),
                None => self.err_ty(),
            };

            param_tys.push(va_list_ty);
        }
        let mut param_tys = param_tys.into_iter().chain(iter::repeat(self.table.next_ty_var()));
        if let Some(self_param) = self.body.self_param
            && let Some(ty) = param_tys.next()
        {
            let ty = self.process_user_written_ty(ty);
            self.write_binding_ty(self_param, ty);
        }
        for (ty, pat) in param_tys.zip(&*self.body.params) {
            let ty = self.process_user_written_ty(ty);

            self.infer_top_pat(*pat, ty, None);
        }
        self.return_ty = match data.ret_type {
            Some(return_ty) => {
                let return_ty = self.with_ty_lowering(
                    &data.store,
                    InferenceTyDiagnosticSource::Signature,
                    LifetimeElisionKind::for_fn_ret(self.interner()),
                    |ctx| {
                        ctx.impl_trait_mode(ImplTraitLoweringMode::Opaque);
                        ctx.lower_ty(return_ty)
                    },
                );
                self.process_user_written_ty(return_ty)
            }
            None => self.types.unit,
        };

        self.return_coercion = Some(CoerceMany::new(self.return_ty));
    }

    #[inline]
    pub(crate) fn interner(&self) -> DbInterner<'db> {
        self.table.interner()
    }

    fn infer_body(&mut self) {
        match self.return_coercion {
            Some(_) => self.infer_return(self.body.body_expr),
            None => {
                _ = self.infer_expr_coerce(
                    self.body.body_expr,
                    &Expectation::has_type(self.return_ty),
                    ExprIsRead::Yes,
                )
            }
        }
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty<'db>) {
        self.result.type_of_expr.insert(expr, ty);
    }

    fn write_expr_adj(&mut self, expr: ExprId, adjustments: Box<[Adjustment<'db>]>) {
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
                        *target = *new_target;
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

    fn write_pat_adj(&mut self, pat: PatId, adjustments: Box<[Ty<'db>]>) {
        if adjustments.is_empty() {
            return;
        }
        self.result.pat_adjustments.entry(pat).or_default().extend(adjustments);
    }

    fn write_method_resolution(&mut self, expr: ExprId, func: FunctionId, subst: GenericArgs<'db>) {
        self.result.method_resolutions.insert(expr, (func, subst));
    }

    fn write_variant_resolution(&mut self, id: ExprOrPatId, variant: VariantId) {
        self.result.variant_resolutions.insert(id, variant);
    }

    fn write_assoc_resolution(
        &mut self,
        id: ExprOrPatId,
        item: AssocItemId,
        subs: GenericArgs<'db>,
    ) {
        self.result.assoc_resolutions.insert(id, (item, subs));
    }

    fn write_pat_ty(&mut self, pat: PatId, ty: Ty<'db>) {
        self.result.type_of_pat.insert(pat, ty);
    }

    fn write_binding_ty(&mut self, id: BindingId, ty: Ty<'db>) {
        self.result.type_of_binding.insert(id, ty);
    }

    fn push_diagnostic(&self, diagnostic: InferenceDiagnostic<'db>) {
        self.diagnostics.push(diagnostic);
    }

    fn with_ty_lowering<R>(
        &mut self,
        store: &ExpressionStore,
        types_source: InferenceTyDiagnosticSource,
        lifetime_elision: LifetimeElisionKind<'db>,
        f: impl FnOnce(&mut TyLoweringContext<'db, '_>) -> R,
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

    fn with_body_ty_lowering<R>(
        &mut self,
        f: impl FnOnce(&mut TyLoweringContext<'db, '_>) -> R,
    ) -> R {
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
        lifetime_elision: LifetimeElisionKind<'db>,
    ) -> Ty<'db> {
        let ty = self
            .with_ty_lowering(store, type_source, lifetime_elision, |ctx| ctx.lower_ty(type_ref));
        self.process_user_written_ty(ty)
    }

    fn make_body_ty(&mut self, type_ref: TypeRefId) -> Ty<'db> {
        self.make_ty(
            type_ref,
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
        )
    }

    fn make_body_const(&mut self, const_ref: ConstRef, ty: Ty<'db>) -> Const<'db> {
        let const_ = self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            |ctx| ctx.lower_const(const_ref, ty),
        );
        self.insert_type_vars(const_)
    }

    fn make_path_as_body_const(&mut self, path: &Path, ty: Ty<'db>) -> Const<'db> {
        let const_ = self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            |ctx| ctx.lower_path_as_const(path, ty),
        );
        self.insert_type_vars(const_)
    }

    fn err_ty(&self) -> Ty<'db> {
        self.types.error
    }

    fn make_body_lifetime(&mut self, lifetime_ref: LifetimeRefId) -> Region<'db> {
        let lt = self.with_ty_lowering(
            self.body,
            InferenceTyDiagnosticSource::Body,
            LifetimeElisionKind::Infer,
            |ctx| ctx.lower_lifetime(lifetime_ref),
        );
        self.insert_type_vars(lt)
    }

    /// Replaces `Ty::Error` by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty<'db>) -> Ty<'db> {
        self.table.insert_type_vars_shallow(ty)
    }

    fn insert_type_vars<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.table.insert_type_vars(ty)
    }

    fn unify(&mut self, ty1: Ty<'db>, ty2: Ty<'db>) -> bool {
        self.table.unify(ty1, ty2)
    }

    /// Attempts to returns the deeply last field of nested structures, but
    /// does not apply any normalization in its search. Returns the same type
    /// if input `ty` is not a structure at all.
    fn struct_tail_without_normalization(&mut self, ty: Ty<'db>) -> Ty<'db> {
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
        mut ty: Ty<'db>,
        mut normalize: impl FnMut(Ty<'db>) -> Ty<'db>,
    ) -> Ty<'db> {
        // FIXME: fetch the limit properly
        let recursion_limit = 10;
        for iteration in 0.. {
            if iteration > recursion_limit {
                return self.err_ty();
            }
            match ty.kind() {
                TyKind::Adt(adt_def, substs) => match adt_def.def_id().0 {
                    AdtId::StructId(struct_id) => {
                        match self.db.field_types(struct_id.into()).values().next_back().copied() {
                            Some(field) => {
                                ty = field.instantiate(self.interner(), substs);
                            }
                            None => break,
                        }
                    }
                    _ => break,
                },
                TyKind::Tuple(substs) => match substs.as_slice().split_last() {
                    Some((last_ty, _)) => ty = *last_ty,
                    None => break,
                },
                TyKind::Alias(..) => {
                    let normalized = normalize(ty);
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

    /// Whenever you lower a user-written type, you should call this.
    fn process_user_written_ty<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.table.process_user_written_ty(ty)
    }

    /// The difference of this method from `process_user_written_ty()` is that this method doesn't register a well-formed obligation,
    /// while `process_user_written_ty()` should (but doesn't currently).
    fn process_remote_user_written_ty<T>(&mut self, ty: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.table.process_remote_user_written_ty(ty)
    }

    fn shallow_resolve(&self, ty: Ty<'db>) -> Ty<'db> {
        self.table.shallow_resolve(ty)
    }

    fn resolve_associated_type(
        &mut self,
        inner_ty: Ty<'db>,
        assoc_ty: Option<TypeAliasId>,
    ) -> Ty<'db> {
        self.resolve_associated_type_with_params(inner_ty, assoc_ty, &[])
    }

    fn demand_eqtype(&mut self, expected: Ty<'db>, actual: Ty<'db>) {
        let result = self
            .table
            .infer_ctxt
            .at(&ObligationCause::new(), self.table.trait_env.env)
            .eq(expected, actual)
            .map(|infer_ok| self.table.register_infer_ok(infer_ok));
        if let Err(_err) = result {
            // FIXME: Emit diagnostic.
        }
    }

    fn resolve_associated_type_with_params(
        &mut self,
        inner_ty: Ty<'db>,
        assoc_ty: Option<TypeAliasId>,
        // FIXME(GATs): these are args for the trait ref, args for assoc type itself should be
        // handled when we support them.
        params: &[GenericArg<'db>],
    ) -> Ty<'db> {
        match assoc_ty {
            Some(res_assoc_ty) => {
                let alias = Ty::new_alias(
                    self.interner(),
                    AliasTyKind::Projection,
                    AliasTy::new(
                        self.interner(),
                        res_assoc_ty.into(),
                        iter::once(inner_ty.into()).chain(params.iter().copied()),
                    ),
                );
                self.table.try_structurally_resolve_type(alias)
            }
            None => self.err_ty(),
        }
    }

    fn resolve_variant(
        &mut self,
        node: ExprOrPatId,
        path: Option<&Path>,
        value_ns: bool,
    ) -> (Ty<'db>, Option<VariantId>) {
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
        let interner = DbInterner::conjure();
        let (resolution, unresolved) = if value_ns {
            let Some(res) = path_ctx.resolve_path_in_value_ns(HygieneId::ROOT) else {
                return (self.err_ty(), None);
            };
            match res {
                ResolveValueResult::ValueNs(value, _) => match value {
                    ValueNs::EnumVariantId(var) => {
                        let args = path_ctx.substs_from_path(var.into(), true, false);
                        drop(ctx);
                        let ty = self
                            .db
                            .ty(var.lookup(self.db).parent.into())
                            .instantiate(interner, args);
                        let ty = self.insert_type_vars(ty);
                        return (ty, Some(var.into()));
                    }
                    ValueNs::StructId(strukt) => {
                        let args = path_ctx.substs_from_path(strukt.into(), true, false);
                        drop(ctx);
                        let ty = self.db.ty(strukt.into()).instantiate(interner, args);
                        let ty = self.insert_type_vars(ty);
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
                let args = path_ctx.substs_from_path(strukt.into(), true, false);
                drop(ctx);
                let ty = self.db.ty(strukt.into()).instantiate(interner, args);
                let ty = self.insert_type_vars(ty);
                forbid_unresolved_segments(self, (ty, Some(strukt.into())), unresolved)
            }
            TypeNs::AdtId(AdtId::UnionId(u)) => {
                let args = path_ctx.substs_from_path(u.into(), true, false);
                drop(ctx);
                let ty = self.db.ty(u.into()).instantiate(interner, args);
                let ty = self.insert_type_vars(ty);
                forbid_unresolved_segments(self, (ty, Some(u.into())), unresolved)
            }
            TypeNs::EnumVariantId(var) => {
                let args = path_ctx.substs_from_path(var.into(), true, false);
                drop(ctx);
                let ty = self.db.ty(var.lookup(self.db).parent.into()).instantiate(interner, args);
                let ty = self.insert_type_vars(ty);
                forbid_unresolved_segments(self, (ty, Some(var.into())), unresolved)
            }
            TypeNs::SelfType(impl_id) => {
                let mut ty = self.db.impl_self_ty(impl_id).instantiate_identity();

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
                    if let TyKind::Adt(adt_def, _) = ty.kind()
                        && let AdtId::EnumId(id) = adt_def.def_id().0
                    {
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
                    ty = self.table.structurally_resolve_type(ty);
                    if ty.is_ty_error() {
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
                let args = path_ctx.substs_from_path_segment(it.into(), true, None, false);
                drop(ctx);
                let interner = DbInterner::conjure();
                let ty = self.db.ty(it.into()).instantiate(interner, args);
                let ty = self.insert_type_vars(ty);

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
            | TypeNs::ModuleId(_) => {
                // FIXME diagnostic
                (self.err_ty(), None)
            }
        };

        fn forbid_unresolved_segments<'db>(
            ctx: &InferenceContext<'_, 'db>,
            result: (Ty<'db>, Option<VariantId>),
            unresolved: Option<usize>,
        ) -> (Ty<'db>, Option<VariantId>) {
            if unresolved.is_none() {
                result
            } else {
                // FIXME diagnostic
                (ctx.types.error, None)
            }
        }
    }

    fn resolve_variant_on_alias(
        &mut self,
        ty: Ty<'db>,
        unresolved: Option<usize>,
        path: &ModPath,
    ) -> (Ty<'db>, Option<VariantId>) {
        let remaining = unresolved.map(|it| path.segments()[it..].len()).filter(|it| it > &0);
        let ty = self.table.try_structurally_resolve_type(ty);
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

    fn get_traits_in_scope<'a>(
        resolver: &Resolver<'db>,
        traits_in_scope: &'a FxHashSet<TraitId>,
    ) -> Either<FxHashSet<TraitId>, &'a FxHashSet<TraitId>> {
        let mut b_traits = resolver.traits_in_scope_from_block_scopes().peekable();
        if b_traits.peek().is_some() {
            Either::Left(traits_in_scope.iter().copied().chain(b_traits).collect())
        } else {
            Either::Right(traits_in_scope)
        }
    }
}

/// When inferring an expression, we propagate downward whatever type hint we
/// are able in the form of an `Expectation`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Expectation<'db> {
    None,
    HasType(Ty<'db>),
    Castable(Ty<'db>),
    RValueLikeUnsized(Ty<'db>),
}

impl<'db> Expectation<'db> {
    /// The expectation that the type of the expression needs to equal the given
    /// type.
    fn has_type(ty: Ty<'db>) -> Self {
        if ty.is_ty_error() {
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
    fn rvalue_hint(ctx: &mut InferenceContext<'_, 'db>, ty: Ty<'db>) -> Self {
        match ctx.struct_tail_without_normalization(ty).kind() {
            TyKind::Slice(_) | TyKind::Str | TyKind::Dynamic(..) => {
                Expectation::RValueLikeUnsized(ty)
            }
            _ => Expectation::has_type(ty),
        }
    }

    /// This expresses no expectation on the type.
    fn none() -> Self {
        Expectation::None
    }

    fn resolve(&self, table: &mut unify::InferenceTable<'db>) -> Expectation<'db> {
        match self {
            Expectation::None => Expectation::None,
            Expectation::HasType(t) => Expectation::HasType(table.shallow_resolve(*t)),
            Expectation::Castable(t) => Expectation::Castable(table.shallow_resolve(*t)),
            Expectation::RValueLikeUnsized(t) => {
                Expectation::RValueLikeUnsized(table.shallow_resolve(*t))
            }
        }
    }

    fn to_option(&self, table: &mut unify::InferenceTable<'db>) -> Option<Ty<'db>> {
        match self.resolve(table) {
            Expectation::None => None,
            Expectation::HasType(t)
            | Expectation::Castable(t)
            | Expectation::RValueLikeUnsized(t) => Some(t),
        }
    }

    fn only_has_type(&self, table: &mut unify::InferenceTable<'db>) -> Option<Ty<'db>> {
        match self {
            Expectation::HasType(t) => Some(table.shallow_resolve(*t)),
            Expectation::Castable(_) | Expectation::RValueLikeUnsized(_) | Expectation::None => {
                None
            }
        }
    }

    fn coercion_target_type(&self, table: &mut unify::InferenceTable<'db>) -> Ty<'db> {
        self.only_has_type(table).unwrap_or_else(|| table.next_ty_var())
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
    fn adjust_for_branches(&self, table: &mut unify::InferenceTable<'db>) -> Expectation<'db> {
        match *self {
            Expectation::HasType(ety) => {
                let ety = table.structurally_resolve_type(ety);
                if ety.is_ty_var() { Expectation::None } else { Expectation::HasType(ety) }
            }
            Expectation::RValueLikeUnsized(ety) => Expectation::RValueLikeUnsized(ety),
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
