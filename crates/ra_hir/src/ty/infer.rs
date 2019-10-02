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
use std::iter::{repeat, repeat_with};
use std::mem;
use std::ops::Index;
use std::sync::Arc;

use ena::unify::{InPlaceUnificationTable, NoError, UnifyKey, UnifyValue};
use rustc_hash::FxHashMap;

use ra_arena::map::ArenaMap;
use ra_prof::profile;
use test_utils::tested_by;

use super::{
    autoderef, lower, method_resolution, op, primitive,
    traits::{Guidance, Obligation, ProjectionPredicate, Solution},
    ApplicationTy, CallableDef, InEnvironment, ProjectionTy, Substs, TraitEnvironment, TraitRef,
    Ty, TypableDef, TypeCtor, TypeWalk,
};
use crate::{
    adt::VariantDef,
    code_model::TypeAlias,
    db::HirDatabase,
    diagnostics::DiagnosticSink,
    expr::{
        self, Array, BinaryOp, BindingAnnotation, Body, Expr, ExprId, Literal, Pat, PatId,
        RecordFieldPat, Statement, UnaryOp,
    },
    generics::{GenericParams, HasGenericParams},
    lang_item::LangItemTarget,
    name,
    nameres::Namespace,
    path::{known, GenericArg, GenericArgs},
    resolve::{Resolver, TypeNs},
    ty::infer::diagnostics::InferenceDiagnostic,
    type_ref::{Mutability, TypeRef},
    Adt, AssocItem, ConstData, DefWithBody, FnData, Function, HasBody, Name, Path, StructField,
};

mod unify;
mod path;

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

macro_rules! ty_app {
    ($ctor:pat, $param:pat) => {
        Ty::Apply(ApplicationTy { ctor: $ctor, parameters: $param })
    };
    ($ctor:pat) => {
        ty_app!($ctor, _)
    };
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

    fn init_coerce_unsized_map(
        db: &'a D,
        resolver: &Resolver,
    ) -> FxHashMap<(TypeCtor, TypeCtor), usize> {
        let krate = resolver.krate().unwrap();
        let impls = match db.lang_item(krate, "coerce_unsized".into()) {
            Some(LangItemTarget::Trait(trait_)) => db.impls_for_trait(krate, trait_),
            _ => return FxHashMap::default(),
        };

        impls
            .iter()
            .filter_map(|impl_block| {
                // `CoerseUnsized` has one generic parameter for the target type.
                let trait_ref = impl_block.target_trait_ref(db)?;
                let cur_from_ty = trait_ref.substs.0.get(0)?;
                let cur_to_ty = trait_ref.substs.0.get(1)?;

                match (&cur_from_ty, cur_to_ty) {
                    (ty_app!(ctor1, st1), ty_app!(ctor2, st2)) => {
                        // FIXME: We return the first non-equal bound as the type parameter to coerce to unsized type.
                        // This works for smart-pointer-like coercion, which covers all impls from std.
                        st1.iter().zip(st2.iter()).enumerate().find_map(|(i, (ty1, ty2))| {
                            match (ty1, ty2) {
                                (Ty::Param { idx: p1, .. }, Ty::Param { idx: p2, .. })
                                    if p1 != p2 =>
                                {
                                    Some(((*ctor1, *ctor2), i))
                                }
                                _ => None,
                            }
                        })
                    }
                    _ => None,
                }
            })
            .collect()
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
                if let Some(known_ty) = self.var_unification_table.probe_value(inner).known() {
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
                    match self.var_unification_table.probe_value(inner).known() {
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
                if let Some(known_ty) = self.var_unification_table.probe_value(inner).known() {
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

    fn infer_tuple_struct_pat(
        &mut self,
        path: Option<&Path>,
        subpats: &[PatId],
        expected: &Ty,
        default_bm: BindingMode,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path);

        self.unify(&ty, expected);

        let substs = ty.substs().unwrap_or_else(Substs::empty);

        for (i, &subpat) in subpats.iter().enumerate() {
            let expected_ty = def
                .and_then(|d| d.field(self.db, &Name::new_tuple_field(i)))
                .map_or(Ty::Unknown, |field| field.ty(self.db))
                .subst(&substs);
            let expected_ty = self.normalize_associated_types_in(expected_ty);
            self.infer_pat(subpat, &expected_ty, default_bm);
        }

        ty
    }

    fn infer_record_pat(
        &mut self,
        path: Option<&Path>,
        subpats: &[RecordFieldPat],
        expected: &Ty,
        default_bm: BindingMode,
        id: PatId,
    ) -> Ty {
        let (ty, def) = self.resolve_variant(path);
        if let Some(variant) = def {
            self.write_variant_resolution(id.into(), variant);
        }

        self.unify(&ty, expected);

        let substs = ty.substs().unwrap_or_else(Substs::empty);

        for subpat in subpats {
            let matching_field = def.and_then(|it| it.field(self.db, &subpat.name));
            let expected_ty =
                matching_field.map_or(Ty::Unknown, |field| field.ty(self.db)).subst(&substs);
            let expected_ty = self.normalize_associated_types_in(expected_ty);
            self.infer_pat(subpat.pat, &expected_ty, default_bm);
        }

        ty
    }

    fn infer_pat(&mut self, pat: PatId, mut expected: &Ty, mut default_bm: BindingMode) -> Ty {
        let body = Arc::clone(&self.body); // avoid borrow checker problem

        let is_non_ref_pat = match &body[pat] {
            Pat::Tuple(..)
            | Pat::TupleStruct { .. }
            | Pat::Record { .. }
            | Pat::Range { .. }
            | Pat::Slice { .. } => true,
            // FIXME: Path/Lit might actually evaluate to ref, but inference is unimplemented.
            Pat::Path(..) | Pat::Lit(..) => true,
            Pat::Wild | Pat::Bind { .. } | Pat::Ref { .. } | Pat::Missing => false,
        };
        if is_non_ref_pat {
            while let Some((inner, mutability)) = expected.as_reference() {
                expected = inner;
                default_bm = match default_bm {
                    BindingMode::Move => BindingMode::Ref(mutability),
                    BindingMode::Ref(Mutability::Shared) => BindingMode::Ref(Mutability::Shared),
                    BindingMode::Ref(Mutability::Mut) => BindingMode::Ref(mutability),
                }
            }
        } else if let Pat::Ref { .. } = &body[pat] {
            tested_by!(match_ergonomics_ref);
            // When you encounter a `&pat` pattern, reset to Move.
            // This is so that `w` is by value: `let (_, &w) = &(1, &2);`
            default_bm = BindingMode::Move;
        }

        // Lose mutability.
        let default_bm = default_bm;
        let expected = expected;

        let ty = match &body[pat] {
            Pat::Tuple(ref args) => {
                let expectations = match expected.as_tuple() {
                    Some(parameters) => &*parameters.0,
                    _ => &[],
                };
                let expectations_iter = expectations.iter().chain(repeat(&Ty::Unknown));

                let inner_tys = args
                    .iter()
                    .zip(expectations_iter)
                    .map(|(&pat, ty)| self.infer_pat(pat, ty, default_bm))
                    .collect();

                Ty::apply(TypeCtor::Tuple { cardinality: args.len() as u16 }, Substs(inner_tys))
            }
            Pat::Ref { pat, mutability } => {
                let expectation = match expected.as_reference() {
                    Some((inner_ty, exp_mut)) => {
                        if *mutability != exp_mut {
                            // FIXME: emit type error?
                        }
                        inner_ty
                    }
                    _ => &Ty::Unknown,
                };
                let subty = self.infer_pat(*pat, expectation, default_bm);
                Ty::apply_one(TypeCtor::Ref(*mutability), subty)
            }
            Pat::TupleStruct { path: p, args: subpats } => {
                self.infer_tuple_struct_pat(p.as_ref(), subpats, expected, default_bm)
            }
            Pat::Record { path: p, args: fields } => {
                self.infer_record_pat(p.as_ref(), fields, expected, default_bm, pat)
            }
            Pat::Path(path) => {
                // FIXME use correct resolver for the surrounding expression
                let resolver = self.resolver.clone();
                self.infer_path(&resolver, &path, pat.into()).unwrap_or(Ty::Unknown)
            }
            Pat::Bind { mode, name: _, subpat } => {
                let mode = if mode == &BindingAnnotation::Unannotated {
                    default_bm
                } else {
                    BindingMode::convert(*mode)
                };
                let inner_ty = if let Some(subpat) = subpat {
                    self.infer_pat(*subpat, expected, default_bm)
                } else {
                    expected.clone()
                };
                let inner_ty = self.insert_type_vars_shallow(inner_ty);

                let bound_ty = match mode {
                    BindingMode::Ref(mutability) => {
                        Ty::apply_one(TypeCtor::Ref(mutability), inner_ty.clone())
                    }
                    BindingMode::Move => inner_ty.clone(),
                };
                let bound_ty = self.resolve_ty_as_possible(&mut vec![], bound_ty);
                self.write_pat_ty(pat, bound_ty);
                return inner_ty;
            }
            _ => Ty::Unknown,
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.unify(&ty, expected);
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        self.write_pat_ty(pat, ty.clone());
        ty
    }

    fn substs_for_method_call(
        &mut self,
        def_generics: Option<Arc<GenericParams>>,
        generic_args: Option<&GenericArgs>,
        receiver_ty: &Ty,
    ) -> Substs {
        let (parent_param_count, param_count) =
            def_generics.as_ref().map_or((0, 0), |g| (g.count_parent_params(), g.params.len()));
        let mut substs = Vec::with_capacity(parent_param_count + param_count);
        // Parent arguments are unknown, except for the receiver type
        if let Some(parent_generics) = def_generics.and_then(|p| p.parent_params.clone()) {
            for param in &parent_generics.params {
                if param.name == name::SELF_TYPE {
                    substs.push(receiver_ty.clone());
                } else {
                    substs.push(Ty::Unknown);
                }
            }
        }
        // handle provided type arguments
        if let Some(generic_args) = generic_args {
            // if args are provided, it should be all of them, but we can't rely on that
            for arg in generic_args.args.iter().take(param_count) {
                match arg {
                    GenericArg::Type(type_ref) => {
                        let ty = self.make_ty(type_ref);
                        substs.push(ty);
                    }
                }
            }
        };
        let supplied_params = substs.len();
        for _ in supplied_params..parent_param_count + param_count {
            substs.push(Ty::Unknown);
        }
        assert_eq!(substs.len(), parent_param_count + param_count);
        Substs(substs.into())
    }

    fn register_obligations_for_call(&mut self, callable_ty: &Ty) {
        if let Ty::Apply(a_ty) = callable_ty {
            if let TypeCtor::FnDef(def) = a_ty.ctor {
                let generic_predicates = self.db.generic_predicates(def.into());
                for predicate in generic_predicates.iter() {
                    let predicate = predicate.clone().subst(&a_ty.parameters);
                    if let Some(obligation) = Obligation::from_predicate(predicate) {
                        self.obligations.push(obligation);
                    }
                }
                // add obligation for trait implementation, if this is a trait method
                match def {
                    CallableDef::Function(f) => {
                        if let Some(trait_) = f.parent_trait(self.db) {
                            // construct a TraitDef
                            let substs = a_ty.parameters.prefix(
                                trait_.generic_params(self.db).count_params_including_parent(),
                            );
                            self.obligations.push(Obligation::Trait(TraitRef { trait_, substs }));
                        }
                    }
                    CallableDef::Struct(_) | CallableDef::EnumVariant(_) => {}
                }
            }
        }
    }

    fn infer_method_call(
        &mut self,
        tgt_expr: ExprId,
        receiver: ExprId,
        args: &[ExprId],
        method_name: &Name,
        generic_args: Option<&GenericArgs>,
    ) -> Ty {
        let receiver_ty = self.infer_expr(receiver, &Expectation::none());
        let canonicalized_receiver = self.canonicalizer().canonicalize_ty(receiver_ty.clone());
        let resolved = method_resolution::lookup_method(
            &canonicalized_receiver.value,
            self.db,
            method_name,
            &self.resolver,
        );
        let (derefed_receiver_ty, method_ty, def_generics) = match resolved {
            Some((ty, func)) => {
                let ty = canonicalized_receiver.decanonicalize_ty(ty);
                self.write_method_resolution(tgt_expr, func);
                (
                    ty,
                    self.db.type_for_def(func.into(), Namespace::Values),
                    Some(func.generic_params(self.db)),
                )
            }
            None => (receiver_ty, Ty::Unknown, None),
        };
        let substs = self.substs_for_method_call(def_generics, generic_args, &derefed_receiver_ty);
        let method_ty = method_ty.apply_substs(substs);
        let method_ty = self.insert_type_vars(method_ty);
        self.register_obligations_for_call(&method_ty);
        let (expected_receiver_ty, param_tys, ret_ty) = match method_ty.callable_sig(self.db) {
            Some(sig) => {
                if !sig.params().is_empty() {
                    (sig.params()[0].clone(), sig.params()[1..].to_vec(), sig.ret().clone())
                } else {
                    (Ty::Unknown, Vec::new(), sig.ret().clone())
                }
            }
            None => (Ty::Unknown, Vec::new(), Ty::Unknown),
        };
        // Apply autoref so the below unification works correctly
        // FIXME: return correct autorefs from lookup_method
        let actual_receiver_ty = match expected_receiver_ty.as_reference() {
            Some((_, mutability)) => Ty::apply_one(TypeCtor::Ref(mutability), derefed_receiver_ty),
            _ => derefed_receiver_ty,
        };
        self.unify(&expected_receiver_ty, &actual_receiver_ty);

        self.check_call_arguments(args, &param_tys);
        let ret_ty = self.normalize_associated_types_in(ret_ty);
        ret_ty
    }

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    fn infer_expr_coerce(&mut self, expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(expr, &expected);
        let ty = if !self.coerce(&ty, &expected.ty) {
            self.result
                .type_mismatches
                .insert(expr, TypeMismatch { expected: expected.ty.clone(), actual: ty.clone() });
            // Return actual type when type mismatch.
            // This is needed for diagnostic when return type mismatch.
            ty
        } else if expected.ty == Ty::Unknown {
            ty
        } else {
            expected.ty.clone()
        };

        self.resolve_ty_as_possible(&mut vec![], ty)
    }

    /// Merge two types from different branches, with possible implicit coerce.
    ///
    /// Note that it is only possible that one type are coerced to another.
    /// Coercing both types to another least upper bound type is not possible in rustc,
    /// which will simply result in "incompatible types" error.
    fn coerce_merge_branch<'t>(&mut self, ty1: &Ty, ty2: &Ty) -> Ty {
        if self.coerce(ty1, ty2) {
            ty2.clone()
        } else if self.coerce(ty2, ty1) {
            ty1.clone()
        } else {
            tested_by!(coerce_merge_fail_fallback);
            // For incompatible types, we use the latter one as result
            // to be better recovery for `if` without `else`.
            ty2.clone()
        }
    }

    /// Unify two types, but may coerce the first one to the second one
    /// using "implicit coercion rules" if needed.
    ///
    /// See: https://doc.rust-lang.org/nomicon/coercions.html
    fn coerce(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let from_ty = self.resolve_ty_shallow(from_ty).into_owned();
        let to_ty = self.resolve_ty_shallow(to_ty);
        self.coerce_inner(from_ty, &to_ty)
    }

    fn coerce_inner(&mut self, mut from_ty: Ty, to_ty: &Ty) -> bool {
        match (&from_ty, to_ty) {
            // Never type will make type variable to fallback to Never Type instead of Unknown.
            (ty_app!(TypeCtor::Never), Ty::Infer(InferTy::TypeVar(tv))) => {
                let var = self.new_maybe_never_type_var();
                self.var_unification_table.union_value(*tv, TypeVarValue::Known(var));
                return true;
            }
            (ty_app!(TypeCtor::Never), _) => return true,

            // Trivial cases, this should go after `never` check to
            // avoid infer result type to be never
            _ => {
                if self.unify_inner_trivial(&from_ty, &to_ty) {
                    return true;
                }
            }
        }

        // Pointer weakening and function to pointer
        match (&mut from_ty, to_ty) {
            // `*mut T`, `&mut T, `&T`` -> `*const T`
            // `&mut T` -> `&T`
            // `&mut T` -> `*mut T`
            (ty_app!(c1@TypeCtor::RawPtr(_)), ty_app!(c2@TypeCtor::RawPtr(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(_)), ty_app!(c2@TypeCtor::RawPtr(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(_)), ty_app!(c2@TypeCtor::Ref(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(Mutability::Mut)), ty_app!(c2@TypeCtor::RawPtr(_))) => {
                *c1 = *c2;
            }

            // Illegal mutablity conversion
            (
                ty_app!(TypeCtor::RawPtr(Mutability::Shared)),
                ty_app!(TypeCtor::RawPtr(Mutability::Mut)),
            )
            | (
                ty_app!(TypeCtor::Ref(Mutability::Shared)),
                ty_app!(TypeCtor::Ref(Mutability::Mut)),
            ) => return false,

            // `{function_type}` -> `fn()`
            (ty_app!(TypeCtor::FnDef(_)), ty_app!(TypeCtor::FnPtr { .. })) => {
                match from_ty.callable_sig(self.db) {
                    None => return false,
                    Some(sig) => {
                        let num_args = sig.params_and_return.len() as u16 - 1;
                        from_ty =
                            Ty::apply(TypeCtor::FnPtr { num_args }, Substs(sig.params_and_return));
                    }
                }
            }

            _ => {}
        }

        if let Some(ret) = self.try_coerce_unsized(&from_ty, &to_ty) {
            return ret;
        }

        // Auto Deref if cannot coerce
        match (&from_ty, to_ty) {
            // FIXME: DerefMut
            (ty_app!(TypeCtor::Ref(_), st1), ty_app!(TypeCtor::Ref(_), st2)) => {
                self.unify_autoderef_behind_ref(&st1[0], &st2[0])
            }

            // Otherwise, normal unify
            _ => self.unify(&from_ty, to_ty),
        }
    }

    /// Coerce a type using `from_ty: CoerceUnsized<ty_ty>`
    ///
    /// See: https://doc.rust-lang.org/nightly/std/marker/trait.CoerceUnsized.html
    fn try_coerce_unsized(&mut self, from_ty: &Ty, to_ty: &Ty) -> Option<bool> {
        let (ctor1, st1, ctor2, st2) = match (from_ty, to_ty) {
            (ty_app!(ctor1, st1), ty_app!(ctor2, st2)) => (ctor1, st1, ctor2, st2),
            _ => return None,
        };

        let coerce_generic_index = *self.coerce_unsized_map.get(&(*ctor1, *ctor2))?;

        // Check `Unsize` first
        match self.check_unsize_and_coerce(
            st1.0.get(coerce_generic_index)?,
            st2.0.get(coerce_generic_index)?,
            0,
        ) {
            Some(true) => {}
            ret => return ret,
        }

        let ret = st1
            .iter()
            .zip(st2.iter())
            .enumerate()
            .filter(|&(idx, _)| idx != coerce_generic_index)
            .all(|(_, (ty1, ty2))| self.unify(ty1, ty2));

        Some(ret)
    }

    /// Check if `from_ty: Unsize<to_ty>`, and coerce to `to_ty` if it holds.
    ///
    /// It should not be directly called. It is only used by `try_coerce_unsized`.
    ///
    /// See: https://doc.rust-lang.org/nightly/std/marker/trait.Unsize.html
    fn check_unsize_and_coerce(&mut self, from_ty: &Ty, to_ty: &Ty, depth: usize) -> Option<bool> {
        if depth > 1000 {
            panic!("Infinite recursion in coercion");
        }

        match (&from_ty, &to_ty) {
            // `[T; N]` -> `[T]`
            (ty_app!(TypeCtor::Array, st1), ty_app!(TypeCtor::Slice, st2)) => {
                Some(self.unify(&st1[0], &st2[0]))
            }

            // `T` -> `dyn Trait` when `T: Trait`
            (_, Ty::Dyn(_)) => {
                // FIXME: Check predicates
                Some(true)
            }

            // `(..., T)` -> `(..., U)` when `T: Unsize<U>`
            (
                ty_app!(TypeCtor::Tuple { cardinality: len1 }, st1),
                ty_app!(TypeCtor::Tuple { cardinality: len2 }, st2),
            ) => {
                if len1 != len2 || *len1 == 0 {
                    return None;
                }

                match self.check_unsize_and_coerce(
                    st1.last().unwrap(),
                    st2.last().unwrap(),
                    depth + 1,
                ) {
                    Some(true) => {}
                    ret => return ret,
                }

                let ret = st1[..st1.len() - 1]
                    .iter()
                    .zip(&st2[..st2.len() - 1])
                    .all(|(ty1, ty2)| self.unify(ty1, ty2));

                Some(ret)
            }

            // Foo<..., T, ...> is Unsize<Foo<..., U, ...>> if:
            // - T: Unsize<U>
            // - Foo is a struct
            // - Only the last field of Foo has a type involving T
            // - T is not part of the type of any other fields
            // - Bar<T>: Unsize<Bar<U>>, if the last field of Foo has type Bar<T>
            (
                ty_app!(TypeCtor::Adt(Adt::Struct(struct1)), st1),
                ty_app!(TypeCtor::Adt(Adt::Struct(struct2)), st2),
            ) if struct1 == struct2 => {
                let fields = struct1.fields(self.db);
                let (last_field, prev_fields) = fields.split_last()?;

                // Get the generic parameter involved in the last field.
                let unsize_generic_index = {
                    let mut index = None;
                    let mut multiple_param = false;
                    last_field.ty(self.db).walk(&mut |ty| match ty {
                        &Ty::Param { idx, .. } => {
                            if index.is_none() {
                                index = Some(idx);
                            } else if Some(idx) != index {
                                multiple_param = true;
                            }
                        }
                        _ => {}
                    });

                    if multiple_param {
                        return None;
                    }
                    index?
                };

                // Check other fields do not involve it.
                let mut multiple_used = false;
                prev_fields.iter().for_each(|field| {
                    field.ty(self.db).walk(&mut |ty| match ty {
                        &Ty::Param { idx, .. } if idx == unsize_generic_index => {
                            multiple_used = true
                        }
                        _ => {}
                    })
                });
                if multiple_used {
                    return None;
                }

                let unsize_generic_index = unsize_generic_index as usize;

                // Check `Unsize` first
                match self.check_unsize_and_coerce(
                    st1.get(unsize_generic_index)?,
                    st2.get(unsize_generic_index)?,
                    depth + 1,
                ) {
                    Some(true) => {}
                    ret => return ret,
                }

                // Then unify other parameters
                let ret = st1
                    .iter()
                    .zip(st2.iter())
                    .enumerate()
                    .filter(|&(idx, _)| idx != unsize_generic_index)
                    .all(|(_, (ty1, ty2))| self.unify(ty1, ty2));

                Some(ret)
            }

            _ => None,
        }
    }

    /// Unify `from_ty` to `to_ty` with optional auto Deref
    ///
    /// Note that the parameters are already stripped the outer reference.
    fn unify_autoderef_behind_ref(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let canonicalized = self.canonicalizer().canonicalize_ty(from_ty.clone());
        let to_ty = self.resolve_ty_shallow(&to_ty);
        // FIXME: Auto DerefMut
        for derefed_ty in
            autoderef::autoderef(self.db, &self.resolver.clone(), canonicalized.value.clone())
        {
            let derefed_ty = canonicalized.decanonicalize_ty(derefed_ty.value);
            match (&*self.resolve_ty_shallow(&derefed_ty), &*to_ty) {
                // Stop when constructor matches.
                (ty_app!(from_ctor, st1), ty_app!(to_ctor, st2)) if from_ctor == to_ctor => {
                    // It will not recurse to `coerce`.
                    return self.unify_substs(st1, st2, 0);
                }
                _ => {}
            }
        }

        false
    }

    fn infer_expr(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(tgt_expr, expected);
        let could_unify = self.unify(&ty, &expected.ty);
        if !could_unify {
            self.result.type_mismatches.insert(
                tgt_expr,
                TypeMismatch { expected: expected.ty.clone(), actual: ty.clone() },
            );
        }
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        ty
    }

    fn infer_expr_inner(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ty = match &body[tgt_expr] {
            Expr::Missing => Ty::Unknown,
            Expr::If { condition, then_branch, else_branch } => {
                // if let is desugared to match, so this is always simple if
                self.infer_expr(*condition, &Expectation::has_type(Ty::simple(TypeCtor::Bool)));

                let then_ty = self.infer_expr_inner(*then_branch, &expected);
                let else_ty = match else_branch {
                    Some(else_branch) => self.infer_expr_inner(*else_branch, &expected),
                    None => Ty::unit(),
                };

                self.coerce_merge_branch(&then_ty, &else_ty)
            }
            Expr::Block { statements, tail } => self.infer_block(statements, *tail, expected),
            Expr::TryBlock { body } => {
                let _inner = self.infer_expr(*body, expected);
                // FIXME should be std::result::Result<{inner}, _>
                Ty::Unknown
            }
            Expr::Loop { body } => {
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                // FIXME handle break with value
                Ty::simple(TypeCtor::Never)
            }
            Expr::While { condition, body } => {
                // while let is desugared to a match loop, so this is always simple while
                self.infer_expr(*condition, &Expectation::has_type(Ty::simple(TypeCtor::Bool)));
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::For { iterable, body, pat } => {
                let iterable_ty = self.infer_expr(*iterable, &Expectation::none());

                let pat_ty = match self.resolve_into_iter_item() {
                    Some(into_iter_item_alias) => {
                        let pat_ty = self.new_type_var();
                        let projection = ProjectionPredicate {
                            ty: pat_ty.clone(),
                            projection_ty: ProjectionTy {
                                associated_ty: into_iter_item_alias,
                                parameters: Substs::single(iterable_ty),
                            },
                        };
                        self.obligations.push(Obligation::Projection(projection));
                        self.resolve_ty_as_possible(&mut vec![], pat_ty)
                    }
                    None => Ty::Unknown,
                };

                self.infer_pat(*pat, &pat_ty, BindingMode::default());
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::Lambda { body, args, arg_types } => {
                assert_eq!(args.len(), arg_types.len());

                let mut sig_tys = Vec::new();

                for (arg_pat, arg_type) in args.iter().zip(arg_types.iter()) {
                    let expected = if let Some(type_ref) = arg_type {
                        self.make_ty(type_ref)
                    } else {
                        Ty::Unknown
                    };
                    let arg_ty = self.infer_pat(*arg_pat, &expected, BindingMode::default());
                    sig_tys.push(arg_ty);
                }

                // add return type
                let ret_ty = self.new_type_var();
                sig_tys.push(ret_ty.clone());
                let sig_ty = Ty::apply(
                    TypeCtor::FnPtr { num_args: sig_tys.len() as u16 - 1 },
                    Substs(sig_tys.into()),
                );
                let closure_ty = Ty::apply_one(
                    TypeCtor::Closure { def: self.body.owner(), expr: tgt_expr },
                    sig_ty,
                );

                // Eagerly try to relate the closure type with the expected
                // type, otherwise we often won't have enough information to
                // infer the body.
                self.coerce(&closure_ty, &expected.ty);

                self.infer_expr(*body, &Expectation::has_type(ret_ty));
                closure_ty
            }
            Expr::Call { callee, args } => {
                let callee_ty = self.infer_expr(*callee, &Expectation::none());
                let (param_tys, ret_ty) = match callee_ty.callable_sig(self.db) {
                    Some(sig) => (sig.params().to_vec(), sig.ret().clone()),
                    None => {
                        // Not callable
                        // FIXME: report an error
                        (Vec::new(), Ty::Unknown)
                    }
                };
                self.register_obligations_for_call(&callee_ty);
                self.check_call_arguments(args, &param_tys);
                let ret_ty = self.normalize_associated_types_in(ret_ty);
                ret_ty
            }
            Expr::MethodCall { receiver, args, method_name, generic_args } => self
                .infer_method_call(tgt_expr, *receiver, &args, &method_name, generic_args.as_ref()),
            Expr::Match { expr, arms } => {
                let input_ty = self.infer_expr(*expr, &Expectation::none());

                let mut result_ty = self.new_maybe_never_type_var();

                for arm in arms {
                    for &pat in &arm.pats {
                        let _pat_ty = self.infer_pat(pat, &input_ty, BindingMode::default());
                    }
                    if let Some(guard_expr) = arm.guard {
                        self.infer_expr(
                            guard_expr,
                            &Expectation::has_type(Ty::simple(TypeCtor::Bool)),
                        );
                    }

                    let arm_ty = self.infer_expr_inner(arm.expr, &expected);
                    result_ty = self.coerce_merge_branch(&result_ty, &arm_ty);
                }

                result_ty
            }
            Expr::Path(p) => {
                // FIXME this could be more efficient...
                let resolver = expr::resolver_for_expr(self.body.clone(), self.db, tgt_expr);
                self.infer_path(&resolver, p, tgt_expr.into()).unwrap_or(Ty::Unknown)
            }
            Expr::Continue => Ty::simple(TypeCtor::Never),
            Expr::Break { expr } => {
                if let Some(expr) = expr {
                    // FIXME handle break with value
                    self.infer_expr(*expr, &Expectation::none());
                }
                Ty::simple(TypeCtor::Never)
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr(*expr, &Expectation::has_type(self.return_ty.clone()));
                }
                Ty::simple(TypeCtor::Never)
            }
            Expr::RecordLit { path, fields, spread } => {
                let (ty, def_id) = self.resolve_variant(path.as_ref());
                if let Some(variant) = def_id {
                    self.write_variant_resolution(tgt_expr.into(), variant);
                }

                self.unify(&ty, &expected.ty);

                let substs = ty.substs().unwrap_or_else(Substs::empty);
                for (field_idx, field) in fields.iter().enumerate() {
                    let field_ty = def_id
                        .and_then(|it| match it.field(self.db, &field.name) {
                            Some(field) => Some(field),
                            None => {
                                self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                    expr: tgt_expr,
                                    field: field_idx,
                                });
                                None
                            }
                        })
                        .map_or(Ty::Unknown, |field| field.ty(self.db))
                        .subst(&substs);
                    self.infer_expr_coerce(field.expr, &Expectation::has_type(field_ty));
                }
                if let Some(expr) = spread {
                    self.infer_expr(*expr, &Expectation::has_type(ty.clone()));
                }
                ty
            }
            Expr::Field { expr, name } => {
                let receiver_ty = self.infer_expr(*expr, &Expectation::none());
                let canonicalized = self.canonicalizer().canonicalize_ty(receiver_ty);
                let ty = autoderef::autoderef(
                    self.db,
                    &self.resolver.clone(),
                    canonicalized.value.clone(),
                )
                .find_map(|derefed_ty| match canonicalized.decanonicalize_ty(derefed_ty.value) {
                    Ty::Apply(a_ty) => match a_ty.ctor {
                        TypeCtor::Tuple { .. } => name
                            .as_tuple_index()
                            .and_then(|idx| a_ty.parameters.0.get(idx).cloned()),
                        TypeCtor::Adt(Adt::Struct(s)) => s.field(self.db, name).map(|field| {
                            self.write_field_resolution(tgt_expr, field);
                            field.ty(self.db).subst(&a_ty.parameters)
                        }),
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap_or(Ty::Unknown);
                let ty = self.insert_type_vars(ty);
                self.normalize_associated_types_in(ty)
            }
            Expr::Await { expr } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                let ty = match self.resolve_future_future_output() {
                    Some(future_future_output_alias) => {
                        let ty = self.new_type_var();
                        let projection = ProjectionPredicate {
                            ty: ty.clone(),
                            projection_ty: ProjectionTy {
                                associated_ty: future_future_output_alias,
                                parameters: Substs::single(inner_ty),
                            },
                        };
                        self.obligations.push(Obligation::Projection(projection));
                        self.resolve_ty_as_possible(&mut vec![], ty)
                    }
                    None => Ty::Unknown,
                };
                ty
            }
            Expr::Try { expr } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                let ty = match self.resolve_ops_try_ok() {
                    Some(ops_try_ok_alias) => {
                        let ty = self.new_type_var();
                        let projection = ProjectionPredicate {
                            ty: ty.clone(),
                            projection_ty: ProjectionTy {
                                associated_ty: ops_try_ok_alias,
                                parameters: Substs::single(inner_ty),
                            },
                        };
                        self.obligations.push(Obligation::Projection(projection));
                        self.resolve_ty_as_possible(&mut vec![], ty)
                    }
                    None => Ty::Unknown,
                };
                ty
            }
            Expr::Cast { expr, type_ref } => {
                let _inner_ty = self.infer_expr(*expr, &Expectation::none());
                let cast_ty = self.make_ty(type_ref);
                // FIXME check the cast...
                cast_ty
            }
            Expr::Ref { expr, mutability } => {
                let expectation =
                    if let Some((exp_inner, exp_mutability)) = &expected.ty.as_reference() {
                        if *exp_mutability == Mutability::Mut && *mutability == Mutability::Shared {
                            // FIXME: throw type error - expected mut reference but found shared ref,
                            // which cannot be coerced
                        }
                        Expectation::has_type(Ty::clone(exp_inner))
                    } else {
                        Expectation::none()
                    };
                // FIXME reference coercions etc.
                let inner_ty = self.infer_expr(*expr, &expectation);
                Ty::apply_one(TypeCtor::Ref(*mutability), inner_ty)
            }
            Expr::Box { expr } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                if let Some(box_) = self.resolve_boxed_box() {
                    Ty::apply_one(TypeCtor::Adt(box_), inner_ty)
                } else {
                    Ty::Unknown
                }
            }
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                match op {
                    UnaryOp::Deref => {
                        let canonicalized = self.canonicalizer().canonicalize_ty(inner_ty);
                        if let Some(derefed_ty) =
                            autoderef::deref(self.db, &self.resolver, &canonicalized.value)
                        {
                            canonicalized.decanonicalize_ty(derefed_ty.value)
                        } else {
                            Ty::Unknown
                        }
                    }
                    UnaryOp::Neg => {
                        match &inner_ty {
                            Ty::Apply(a_ty) => match a_ty.ctor {
                                TypeCtor::Int(primitive::UncertainIntTy::Unknown)
                                | TypeCtor::Int(primitive::UncertainIntTy::Known(
                                    primitive::IntTy {
                                        signedness: primitive::Signedness::Signed,
                                        ..
                                    },
                                ))
                                | TypeCtor::Float(..) => inner_ty,
                                _ => Ty::Unknown,
                            },
                            Ty::Infer(InferTy::IntVar(..)) | Ty::Infer(InferTy::FloatVar(..)) => {
                                inner_ty
                            }
                            // FIXME: resolve ops::Neg trait
                            _ => Ty::Unknown,
                        }
                    }
                    UnaryOp::Not => {
                        match &inner_ty {
                            Ty::Apply(a_ty) => match a_ty.ctor {
                                TypeCtor::Bool | TypeCtor::Int(_) => inner_ty,
                                _ => Ty::Unknown,
                            },
                            Ty::Infer(InferTy::IntVar(..)) => inner_ty,
                            // FIXME: resolve ops::Not trait for inner_ty
                            _ => Ty::Unknown,
                        }
                    }
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => match op {
                Some(op) => {
                    let lhs_expectation = match op {
                        BinaryOp::LogicOp(..) => Expectation::has_type(Ty::simple(TypeCtor::Bool)),
                        _ => Expectation::none(),
                    };
                    let lhs_ty = self.infer_expr(*lhs, &lhs_expectation);
                    // FIXME: find implementation of trait corresponding to operation
                    // symbol and resolve associated `Output` type
                    let rhs_expectation = op::binary_op_rhs_expectation(*op, lhs_ty);
                    let rhs_ty = self.infer_expr(*rhs, &Expectation::has_type(rhs_expectation));

                    // FIXME: similar as above, return ty is often associated trait type
                    op::binary_op_return_ty(*op, rhs_ty)
                }
                _ => Ty::Unknown,
            },
            Expr::Index { base, index } => {
                let _base_ty = self.infer_expr(*base, &Expectation::none());
                let _index_ty = self.infer_expr(*index, &Expectation::none());
                // FIXME: use `std::ops::Index::Output` to figure out the real return type
                Ty::Unknown
            }
            Expr::Tuple { exprs } => {
                let mut tys = match &expected.ty {
                    ty_app!(TypeCtor::Tuple { .. }, st) => st
                        .iter()
                        .cloned()
                        .chain(repeat_with(|| self.new_type_var()))
                        .take(exprs.len())
                        .collect::<Vec<_>>(),
                    _ => (0..exprs.len()).map(|_| self.new_type_var()).collect(),
                };

                for (expr, ty) in exprs.iter().zip(tys.iter_mut()) {
                    self.infer_expr_coerce(*expr, &Expectation::has_type(ty.clone()));
                }

                Ty::apply(TypeCtor::Tuple { cardinality: tys.len() as u16 }, Substs(tys.into()))
            }
            Expr::Array(array) => {
                let elem_ty = match &expected.ty {
                    ty_app!(TypeCtor::Array, st) | ty_app!(TypeCtor::Slice, st) => {
                        st.as_single().clone()
                    }
                    _ => self.new_type_var(),
                };

                match array {
                    Array::ElementList(items) => {
                        for expr in items.iter() {
                            self.infer_expr_coerce(*expr, &Expectation::has_type(elem_ty.clone()));
                        }
                    }
                    Array::Repeat { initializer, repeat } => {
                        self.infer_expr_coerce(
                            *initializer,
                            &Expectation::has_type(elem_ty.clone()),
                        );
                        self.infer_expr(
                            *repeat,
                            &Expectation::has_type(Ty::simple(TypeCtor::Int(
                                primitive::UncertainIntTy::Known(primitive::IntTy::usize()),
                            ))),
                        );
                    }
                }

                Ty::apply_one(TypeCtor::Array, elem_ty)
            }
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => Ty::simple(TypeCtor::Bool),
                Literal::String(..) => {
                    Ty::apply_one(TypeCtor::Ref(Mutability::Shared), Ty::simple(TypeCtor::Str))
                }
                Literal::ByteString(..) => {
                    let byte_type = Ty::simple(TypeCtor::Int(primitive::UncertainIntTy::Known(
                        primitive::IntTy::u8(),
                    )));
                    let slice_type = Ty::apply_one(TypeCtor::Slice, byte_type);
                    Ty::apply_one(TypeCtor::Ref(Mutability::Shared), slice_type)
                }
                Literal::Char(..) => Ty::simple(TypeCtor::Char),
                Literal::Int(_v, ty) => Ty::simple(TypeCtor::Int(*ty)),
                Literal::Float(_v, ty) => Ty::simple(TypeCtor::Float(*ty)),
            },
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        self.write_expr_ty(tgt_expr, ty.clone());
        ty
    }

    fn infer_block(
        &mut self,
        statements: &[Statement],
        tail: Option<ExprId>,
        expected: &Expectation,
    ) -> Ty {
        let mut diverges = false;
        for stmt in statements {
            match stmt {
                Statement::Let { pat, type_ref, initializer } => {
                    let decl_ty =
                        type_ref.as_ref().map(|tr| self.make_ty(tr)).unwrap_or(Ty::Unknown);

                    // Always use the declared type when specified
                    let mut ty = decl_ty.clone();

                    if let Some(expr) = initializer {
                        let actual_ty =
                            self.infer_expr_coerce(*expr, &Expectation::has_type(decl_ty.clone()));
                        if decl_ty == Ty::Unknown {
                            ty = actual_ty;
                        }
                    }

                    let ty = self.resolve_ty_as_possible(&mut vec![], ty);
                    self.infer_pat(*pat, &ty, BindingMode::default());
                }
                Statement::Expr(expr) => {
                    if let ty_app!(TypeCtor::Never) = self.infer_expr(*expr, &Expectation::none()) {
                        diverges = true;
                    }
                }
            }
        }

        let ty = if let Some(expr) = tail {
            self.infer_expr_coerce(expr, expected)
        } else {
            self.coerce(&Ty::unit(), &expected.ty);
            Ty::unit()
        };
        if diverges {
            Ty::simple(TypeCtor::Never)
        } else {
            ty
        }
    }

    fn check_call_arguments(&mut self, args: &[ExprId], param_tys: &[Ty]) {
        // Quoting https://github.com/rust-lang/rust/blob/6ef275e6c3cb1384ec78128eceeb4963ff788dca/src/librustc_typeck/check/mod.rs#L3325 --
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for &check_closures in &[false, true] {
            let param_iter = param_tys.iter().cloned().chain(repeat(Ty::Unknown));
            for (&arg, param_ty) in args.iter().zip(param_iter) {
                let is_closure = match &self.body[arg] {
                    Expr::Lambda { .. } => true,
                    _ => false,
                };

                if is_closure != check_closures {
                    continue;
                }

                let param_ty = self.normalize_associated_types_in(param_ty);
                self.infer_expr_coerce(arg, &Expectation::has_type(param_ty.clone()));
            }
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
