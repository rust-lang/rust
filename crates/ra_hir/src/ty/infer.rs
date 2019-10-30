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

use std::borrow::Cow;
use std::mem;
use std::ops::Index;
use std::sync::Arc;

use ena::unify::{InPlaceUnificationTable, NoError, UnifyKey, UnifyValue};
use rustc_hash::FxHashMap;

use hir_def::{
    path::known,
    type_ref::{Mutability, TypeRef},
};
use hir_expand::name;
use ra_arena::map::ArenaMap;
use ra_prof::profile;
use test_utils::tested_by;

use super::{
    lower, primitive,
    traits::{Guidance, Obligation, ProjectionPredicate, Solution},
    ApplicationTy, InEnvironment, ProjectionTy, Substs, TraitEnvironment, TraitRef, Ty, TypableDef,
    TypeCtor, TypeWalk,
};
use crate::{
    adt::VariantDef,
    code_model::TypeAlias,
    db::HirDatabase,
    diagnostics::DiagnosticSink,
    expr::{BindingAnnotation, Body, ExprId, PatId},
    resolve::{Resolver, TypeNs},
    ty::infer::diagnostics::InferenceDiagnostic,
    Adt, AssocItem, ConstData, DefWithBody, FnData, Function, HasBody, Path, StructField,
};

macro_rules! ty_app {
    ($ctor:pat, $param:pat) => {
        crate::ty::Ty::Apply(crate::ty::ApplicationTy { ctor: $ctor, parameters: $param })
    };
    ($ctor:pat) => {
        ty_app!($ctor, _)
    };
}

mod unify;
mod path;
mod expr;
mod pat;
mod coerce;

/// The entry point of type inference.
pub fn infer_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<InferenceResult> {
    let _p = profile("infer_query");
    let body = def.body(db);
    let resolver = def.resolver(db);
    let mut ctx = InferenceContext::new(db, body, resolver);

    match def {
        DefWithBody::Const(ref c) => ctx.collect_const(&c.data(db)),
        DefWithBody::Function(ref f) => ctx.collect_fn(&f.data(db)),
        DefWithBody::Static(ref s) => ctx.collect_const(&s.data(db)),
    }

    ctx.infer_body();

    Arc::new(ctx.resolve_all())
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
enum ExprOrPatId {
    ExprId(ExprId),
    PatId(PatId),
}

impl_froms!(ExprOrPatId: ExprId, PatId);

/// Binding modes inferred for patterns.
/// https://doc.rust-lang.org/reference/patterns.html#binding-modes
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BindingMode {
    Move,
    Ref(Mutability),
}

impl BindingMode {
    pub fn convert(annotation: BindingAnnotation) -> BindingMode {
        match annotation {
            BindingAnnotation::Unannotated | BindingAnnotation::Mutable => BindingMode::Move,
            BindingAnnotation::Ref => BindingMode::Ref(Mutability::Shared),
            BindingAnnotation::RefMut => BindingMode::Ref(Mutability::Mut),
        }
    }
}

impl Default for BindingMode {
    fn default() -> Self {
        BindingMode::Move
    }
}

/// A mismatch between an expected and an inferred type.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TypeMismatch {
    pub expected: Ty,
    pub actual: Ty,
}

/// The result of type inference: A mapping from expressions and patterns to types.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct InferenceResult {
    /// For each method call expr, records the function it resolves to.
    method_resolutions: FxHashMap<ExprId, Function>,
    /// For each field access expr, records the field it resolves to.
    field_resolutions: FxHashMap<ExprId, StructField>,
    /// For each struct literal, records the variant it resolves to.
    variant_resolutions: FxHashMap<ExprOrPatId, VariantDef>,
    /// For each associated item record what it resolves to
    assoc_resolutions: FxHashMap<ExprOrPatId, AssocItem>,
    diagnostics: Vec<InferenceDiagnostic>,
    pub(super) type_of_expr: ArenaMap<ExprId, Ty>,
    pub(super) type_of_pat: ArenaMap<PatId, Ty>,
    pub(super) type_mismatches: ArenaMap<ExprId, TypeMismatch>,
}

impl InferenceResult {
    pub fn method_resolution(&self, expr: ExprId) -> Option<Function> {
        self.method_resolutions.get(&expr).copied()
    }
    pub fn field_resolution(&self, expr: ExprId) -> Option<StructField> {
        self.field_resolutions.get(&expr).copied()
    }
    pub fn variant_resolution_for_expr(&self, id: ExprId) -> Option<VariantDef> {
        self.variant_resolutions.get(&id.into()).copied()
    }
    pub fn variant_resolution_for_pat(&self, id: PatId) -> Option<VariantDef> {
        self.variant_resolutions.get(&id.into()).copied()
    }
    pub fn assoc_resolutions_for_expr(&self, id: ExprId) -> Option<AssocItem> {
        self.assoc_resolutions.get(&id.into()).copied()
    }
    pub fn assoc_resolutions_for_pat(&self, id: PatId) -> Option<AssocItem> {
        self.assoc_resolutions.get(&id.into()).copied()
    }
    pub fn type_mismatch_for_expr(&self, expr: ExprId) -> Option<&TypeMismatch> {
        self.type_mismatches.get(expr)
    }
    pub(crate) fn add_diagnostics(
        &self,
        db: &impl HirDatabase,
        owner: Function,
        sink: &mut DiagnosticSink,
    ) {
        self.diagnostics.iter().for_each(|it| it.add_to(db, owner, sink))
    }
}

impl Index<ExprId> for InferenceResult {
    type Output = Ty;

    fn index(&self, expr: ExprId) -> &Ty {
        self.type_of_expr.get(expr).unwrap_or(&Ty::Unknown)
    }
}

impl Index<PatId> for InferenceResult {
    type Output = Ty;

    fn index(&self, pat: PatId) -> &Ty {
        self.type_of_pat.get(pat).unwrap_or(&Ty::Unknown)
    }
}

/// The inference context contains all information needed during type inference.
#[derive(Clone, Debug)]
struct InferenceContext<'a, D: HirDatabase> {
    db: &'a D,
    body: Arc<Body>,
    resolver: Resolver,
    var_unification_table: InPlaceUnificationTable<TypeVarId>,
    trait_env: Arc<TraitEnvironment>,
    obligations: Vec<Obligation>,
    result: InferenceResult,
    /// The return type of the function being inferred.
    return_ty: Ty,

    /// Impls of `CoerceUnsized` used in coercion.
    /// (from_ty_ctor, to_ty_ctor) => coerce_generic_index
    // FIXME: Use trait solver for this.
    // Chalk seems unable to work well with builtin impl of `Unsize` now.
    coerce_unsized_map: FxHashMap<(TypeCtor, TypeCtor), usize>,
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(db: &'a D, body: Arc<Body>, resolver: Resolver) -> Self {
        InferenceContext {
            result: InferenceResult::default(),
            var_unification_table: InPlaceUnificationTable::new(),
            obligations: Vec::default(),
            return_ty: Ty::Unknown, // set in collect_fn_signature
            trait_env: lower::trait_env(db, &resolver),
            coerce_unsized_map: Self::init_coerce_unsized_map(db, &resolver),
            db,
            body,
            resolver,
        }
    }

    fn resolve_all(mut self) -> InferenceResult {
        // FIXME resolve obligations as well (use Guidance if necessary)
        let mut result = mem::replace(&mut self.result, InferenceResult::default());
        let mut tv_stack = Vec::new();
        for ty in result.type_of_expr.values_mut() {
            let resolved = self.resolve_ty_completely(&mut tv_stack, mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        for ty in result.type_of_pat.values_mut() {
            let resolved = self.resolve_ty_completely(&mut tv_stack, mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        result
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty) {
        self.result.type_of_expr.insert(expr, ty);
    }

    fn write_method_resolution(&mut self, expr: ExprId, func: Function) {
        self.result.method_resolutions.insert(expr, func);
    }

    fn write_field_resolution(&mut self, expr: ExprId, field: StructField) {
        self.result.field_resolutions.insert(expr, field);
    }

    fn write_variant_resolution(&mut self, id: ExprOrPatId, variant: VariantDef) {
        self.result.variant_resolutions.insert(id, variant);
    }

    fn write_assoc_resolution(&mut self, id: ExprOrPatId, item: AssocItem) {
        self.result.assoc_resolutions.insert(id, item);
    }

    fn write_pat_ty(&mut self, pat: PatId, ty: Ty) {
        self.result.type_of_pat.insert(pat, ty);
    }

    fn push_diagnostic(&mut self, diagnostic: InferenceDiagnostic) {
        self.result.diagnostics.push(diagnostic);
    }

    fn make_ty(&mut self, type_ref: &TypeRef) -> Ty {
        let ty = Ty::from_hir(
            self.db,
            // FIXME use right resolver for block
            &self.resolver,
            type_ref,
        );
        let ty = self.insert_type_vars(ty);
        self.normalize_associated_types_in(ty)
    }

    fn unify_substs(&mut self, substs1: &Substs, substs2: &Substs, depth: usize) -> bool {
        substs1.0.iter().zip(substs2.0.iter()).all(|(t1, t2)| self.unify_inner(t1, t2, depth))
    }

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        self.unify_inner(ty1, ty2, 0)
    }

    fn unify_inner(&mut self, ty1: &Ty, ty2: &Ty, depth: usize) -> bool {
        if depth > 1000 {
            // prevent stackoverflows
            panic!("infinite recursion in unification");
        }
        if ty1 == ty2 {
            return true;
        }
        // try to resolve type vars first
        let ty1 = self.resolve_ty_shallow(ty1);
        let ty2 = self.resolve_ty_shallow(ty2);
        match (&*ty1, &*ty2) {
            (Ty::Apply(a_ty1), Ty::Apply(a_ty2)) if a_ty1.ctor == a_ty2.ctor => {
                self.unify_substs(&a_ty1.parameters, &a_ty2.parameters, depth + 1)
            }
            _ => self.unify_inner_trivial(&ty1, &ty2),
        }
    }

    fn unify_inner_trivial(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        match (ty1, ty2) {
            (Ty::Unknown, _) | (_, Ty::Unknown) => true,

            (Ty::Infer(InferTy::TypeVar(tv1)), Ty::Infer(InferTy::TypeVar(tv2)))
            | (Ty::Infer(InferTy::IntVar(tv1)), Ty::Infer(InferTy::IntVar(tv2)))
            | (Ty::Infer(InferTy::FloatVar(tv1)), Ty::Infer(InferTy::FloatVar(tv2)))
            | (
                Ty::Infer(InferTy::MaybeNeverTypeVar(tv1)),
                Ty::Infer(InferTy::MaybeNeverTypeVar(tv2)),
            ) => {
                // both type vars are unknown since we tried to resolve them
                self.var_unification_table.union(*tv1, *tv2);
                true
            }

            // The order of MaybeNeverTypeVar matters here.
            // Unifying MaybeNeverTypeVar and TypeVar will let the latter become MaybeNeverTypeVar.
            // Unifying MaybeNeverTypeVar and other concrete type will let the former become it.
            (Ty::Infer(InferTy::TypeVar(tv)), other)
            | (other, Ty::Infer(InferTy::TypeVar(tv)))
            | (Ty::Infer(InferTy::MaybeNeverTypeVar(tv)), other)
            | (other, Ty::Infer(InferTy::MaybeNeverTypeVar(tv)))
            | (Ty::Infer(InferTy::IntVar(tv)), other @ ty_app!(TypeCtor::Int(_)))
            | (other @ ty_app!(TypeCtor::Int(_)), Ty::Infer(InferTy::IntVar(tv)))
            | (Ty::Infer(InferTy::FloatVar(tv)), other @ ty_app!(TypeCtor::Float(_)))
            | (other @ ty_app!(TypeCtor::Float(_)), Ty::Infer(InferTy::FloatVar(tv))) => {
                // the type var is unknown since we tried to resolve it
                self.var_unification_table.union_value(*tv, TypeVarValue::Known(other.clone()));
                true
            }

            _ => false,
        }
    }

    fn new_type_var(&mut self) -> Ty {
        Ty::Infer(InferTy::TypeVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    fn new_integer_var(&mut self) -> Ty {
        Ty::Infer(InferTy::IntVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    fn new_float_var(&mut self) -> Ty {
        Ty::Infer(InferTy::FloatVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    fn new_maybe_never_type_var(&mut self) -> Ty {
        Ty::Infer(InferTy::MaybeNeverTypeVar(
            self.var_unification_table.new_key(TypeVarValue::Unknown),
        ))
    }

    /// Replaces Ty::Unknown by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty {
            Ty::Unknown => self.new_type_var(),
            Ty::Apply(ApplicationTy {
                ctor: TypeCtor::Int(primitive::UncertainIntTy::Unknown),
                ..
            }) => self.new_integer_var(),
            Ty::Apply(ApplicationTy {
                ctor: TypeCtor::Float(primitive::UncertainFloatTy::Unknown),
                ..
            }) => self.new_float_var(),
            _ => ty,
        }
    }

    fn insert_type_vars(&mut self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| self.insert_type_vars_shallow(ty))
    }

    fn resolve_obligations_as_possible(&mut self) {
        let obligations = mem::replace(&mut self.obligations, Vec::new());
        for obligation in obligations {
            let in_env = InEnvironment::new(self.trait_env.clone(), obligation.clone());
            let canonicalized = self.canonicalizer().canonicalize_obligation(in_env);
            let solution =
                self.db.trait_solve(self.resolver.krate().unwrap(), canonicalized.value.clone());

            match solution {
                Some(Solution::Unique(substs)) => {
                    canonicalized.apply_solution(self, substs.0);
                }
                Some(Solution::Ambig(Guidance::Definite(substs))) => {
                    canonicalized.apply_solution(self, substs.0);
                    self.obligations.push(obligation);
                }
                Some(_) => {
                    // FIXME use this when trying to resolve everything at the end
                    self.obligations.push(obligation);
                }
                None => {
                    // FIXME obligation cannot be fulfilled => diagnostic
                }
            };
        }
    }

    /// Resolves the type as far as currently possible, replacing type variables
    /// by their known types. All types returned by the infer_* functions should
    /// be resolved as far as possible, i.e. contain no type variables with
    /// known type.
    fn resolve_ty_as_possible(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        self.resolve_obligations_as_possible();

        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    tested_by!(type_var_cycles_resolve_as_possible);
                    // recursive type
                    return tv.fallback_value();
                }
                if let Some(known_ty) =
                    self.var_unification_table.inlined_probe_value(inner).known()
                {
                    // known_ty may contain other variables that are known by now
                    tv_stack.push(inner);
                    let result = self.resolve_ty_as_possible(tv_stack, known_ty.clone());
                    tv_stack.pop();
                    result
                } else {
                    ty
                }
            }
            _ => ty,
        })
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        let mut ty = Cow::Borrowed(ty);
        // The type variable could resolve to a int/float variable. Hence try
        // resolving up to three times; each type of variable shouldn't occur
        // more than once
        for i in 0..3 {
            if i > 0 {
                tested_by!(type_var_resolves_to_int_var);
            }
            match &*ty {
                Ty::Infer(tv) => {
                    let inner = tv.to_inner();
                    match self.var_unification_table.inlined_probe_value(inner).known() {
                        Some(known_ty) => {
                            // The known_ty can't be a type var itself
                            ty = Cow::Owned(known_ty.clone());
                        }
                        _ => return ty,
                    }
                }
                _ => return ty,
            }
        }
        log::error!("Inference variable still not resolved: {:?}", ty);
        ty
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    fn normalize_associated_types_in(&mut self, ty: Ty) -> Ty {
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        ty.fold(&mut |ty| match ty {
            Ty::Projection(proj_ty) => self.normalize_projection_ty(proj_ty),
            _ => ty,
        })
    }

    fn normalize_projection_ty(&mut self, proj_ty: ProjectionTy) -> Ty {
        let var = self.new_type_var();
        let predicate = ProjectionPredicate { projection_ty: proj_ty, ty: var.clone() };
        let obligation = Obligation::Projection(predicate);
        self.obligations.push(obligation);
        var
    }

    /// Resolves the type completely; type variables without known type are
    /// replaced by Ty::Unknown.
    fn resolve_ty_completely(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    tested_by!(type_var_cycles_resolve_completely);
                    // recursive type
                    return tv.fallback_value();
                }
                if let Some(known_ty) =
                    self.var_unification_table.inlined_probe_value(inner).known()
                {
                    // known_ty may contain other variables that are known by now
                    tv_stack.push(inner);
                    let result = self.resolve_ty_completely(tv_stack, known_ty.clone());
                    tv_stack.pop();
                    result
                } else {
                    tv.fallback_value()
                }
            }
            _ => ty,
        })
    }

    fn resolve_variant(&mut self, path: Option<&Path>) -> (Ty, Option<VariantDef>) {
        let path = match path {
            Some(path) => path,
            None => return (Ty::Unknown, None),
        };
        let resolver = &self.resolver;
        let def: TypableDef =
            // FIXME: this should resolve assoc items as well, see this example:
            // https://play.rust-lang.org/?gist=087992e9e22495446c01c0d4e2d69521
            match resolver.resolve_path_in_type_ns_fully(self.db, &path) {
                Some(TypeNs::Adt(Adt::Struct(it))) => it.into(),
                Some(TypeNs::Adt(Adt::Union(it))) => it.into(),
                Some(TypeNs::AdtSelfType(adt)) => adt.into(),
                Some(TypeNs::EnumVariant(it)) => it.into(),
                Some(TypeNs::TypeAlias(it)) => it.into(),

                Some(TypeNs::SelfType(_)) |
                Some(TypeNs::GenericParam(_)) |
                Some(TypeNs::BuiltinType(_)) |
                Some(TypeNs::Trait(_)) |
                Some(TypeNs::Adt(Adt::Enum(_))) |
                None => {
                    return (Ty::Unknown, None)
                }
            };
        // FIXME remove the duplication between here and `Ty::from_path`?
        let substs = Ty::substs_from_path(self.db, resolver, path, def);
        match def {
            TypableDef::Adt(Adt::Struct(s)) => {
                let ty = s.ty(self.db);
                let ty = self.insert_type_vars(ty.apply_substs(substs));
                (ty, Some(s.into()))
            }
            TypableDef::EnumVariant(var) => {
                let ty = var.parent_enum(self.db).ty(self.db);
                let ty = self.insert_type_vars(ty.apply_substs(substs));
                (ty, Some(var.into()))
            }
            TypableDef::Adt(Adt::Enum(_))
            | TypableDef::Adt(Adt::Union(_))
            | TypableDef::TypeAlias(_)
            | TypableDef::Function(_)
            | TypableDef::Const(_)
            | TypableDef::Static(_)
            | TypableDef::BuiltinType(_) => (Ty::Unknown, None),
        }
    }

    fn collect_const(&mut self, data: &ConstData) {
        self.return_ty = self.make_ty(data.type_ref());
    }

    fn collect_fn(&mut self, data: &FnData) {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        for (type_ref, pat) in data.params().iter().zip(body.params()) {
            let ty = self.make_ty(type_ref);

            self.infer_pat(*pat, &ty, BindingMode::default());
        }
        self.return_ty = self.make_ty(data.ret_type());
    }

    fn infer_body(&mut self) {
        self.infer_expr(self.body.body_expr(), &Expectation::has_type(self.return_ty.clone()));
    }

    fn resolve_into_iter_item(&self) -> Option<TypeAlias> {
        let path = known::std_iter_into_iterator();
        let trait_ = self.resolver.resolve_known_trait(self.db, &path)?;
        trait_.associated_type_by_name(self.db, &name::ITEM_TYPE)
    }

    fn resolve_ops_try_ok(&self) -> Option<TypeAlias> {
        let path = known::std_ops_try();
        let trait_ = self.resolver.resolve_known_trait(self.db, &path)?;
        trait_.associated_type_by_name(self.db, &name::OK_TYPE)
    }

    fn resolve_future_future_output(&self) -> Option<TypeAlias> {
        let path = known::std_future_future();
        let trait_ = self.resolver.resolve_known_trait(self.db, &path)?;
        trait_.associated_type_by_name(self.db, &name::OUTPUT_TYPE)
    }

    fn resolve_boxed_box(&self) -> Option<Adt> {
        let path = known::std_boxed_box();
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(Adt::Struct(struct_))
    }
}

/// The ID of a type variable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeVarId(pub(super) u32);

impl UnifyKey for TypeVarId {
    type Value = TypeVarValue;

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(i: u32) -> Self {
        TypeVarId(i)
    }

    fn tag() -> &'static str {
        "TypeVarId"
    }
}

/// The value of a type variable: either we already know the type, or we don't
/// know it yet.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeVarValue {
    Known(Ty),
    Unknown,
}

impl TypeVarValue {
    fn known(&self) -> Option<&Ty> {
        match self {
            TypeVarValue::Known(ty) => Some(ty),
            TypeVarValue::Unknown => None,
        }
    }
}

impl UnifyValue for TypeVarValue {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, NoError> {
        match (value1, value2) {
            // We should never equate two type variables, both of which have
            // known types. Instead, we recursively equate those types.
            (TypeVarValue::Known(t1), TypeVarValue::Known(t2)) => panic!(
                "equating two type variables, both of which have known types: {:?} and {:?}",
                t1, t2
            ),

            // If one side is known, prefer that one.
            (TypeVarValue::Known(..), TypeVarValue::Unknown) => Ok(value1.clone()),
            (TypeVarValue::Unknown, TypeVarValue::Known(..)) => Ok(value2.clone()),

            (TypeVarValue::Unknown, TypeVarValue::Unknown) => Ok(TypeVarValue::Unknown),
        }
    }
}

/// The kinds of placeholders we need during type inference. There's separate
/// values for general types, and for integer and float variables. The latter
/// two are used for inference of literal values (e.g. `100` could be one of
/// several integer types).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InferTy {
    TypeVar(TypeVarId),
    IntVar(TypeVarId),
    FloatVar(TypeVarId),
    MaybeNeverTypeVar(TypeVarId),
}

impl InferTy {
    fn to_inner(self) -> TypeVarId {
        match self {
            InferTy::TypeVar(ty)
            | InferTy::IntVar(ty)
            | InferTy::FloatVar(ty)
            | InferTy::MaybeNeverTypeVar(ty) => ty,
        }
    }

    fn fallback_value(self) -> Ty {
        match self {
            InferTy::TypeVar(..) => Ty::Unknown,
            InferTy::IntVar(..) => {
                Ty::simple(TypeCtor::Int(primitive::UncertainIntTy::Known(primitive::IntTy::i32())))
            }
            InferTy::FloatVar(..) => Ty::simple(TypeCtor::Float(
                primitive::UncertainFloatTy::Known(primitive::FloatTy::f64()),
            )),
            InferTy::MaybeNeverTypeVar(..) => Ty::simple(TypeCtor::Never),
        }
    }
}

/// When inferring an expression, we propagate downward whatever type hint we
/// are able in the form of an `Expectation`.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Expectation {
    ty: Ty,
    // FIXME: In some cases, we need to be aware whether the expectation is that
    // the type match exactly what we passed, or whether it just needs to be
    // coercible to the expected type. See Expectation::rvalue_hint in rustc.
}

impl Expectation {
    /// The expectation that the type of the expression needs to equal the given
    /// type.
    fn has_type(ty: Ty) -> Self {
        Expectation { ty }
    }

    /// This expresses no expectation on the type.
    fn none() -> Self {
        Expectation { ty: Ty::Unknown }
    }
}

mod diagnostics {
    use crate::{
        db::HirDatabase,
        diagnostics::{DiagnosticSink, NoSuchField},
        expr::ExprId,
        Function, HasSource,
    };

    #[derive(Debug, PartialEq, Eq, Clone)]
    pub(super) enum InferenceDiagnostic {
        NoSuchField { expr: ExprId, field: usize },
    }

    impl InferenceDiagnostic {
        pub(super) fn add_to(
            &self,
            db: &impl HirDatabase,
            owner: Function,
            sink: &mut DiagnosticSink,
        ) {
            match self {
                InferenceDiagnostic::NoSuchField { expr, field } => {
                    let file = owner.source(db).file_id;
                    let field = owner.body_source_map(db).field_syntax(*expr, *field);
                    sink.push(NoSuchField { file, field })
                }
            }
        }
    }
}
