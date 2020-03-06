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

use rustc_hash::FxHashMap;

use hir_def::{
    body::Body,
    data::{ConstData, FunctionData},
    expr::{BindingAnnotation, ExprId, PatId},
    lang_item::LangItemTarget,
    path::{path, Path},
    resolver::{HasResolver, Resolver, TypeNs},
    type_ref::{Mutability, TypeRef},
    AdtId, AssocItemId, DefWithBodyId, FunctionId, StructFieldId, TraitId, TypeAliasId, VariantId,
};
use hir_expand::{diagnostics::DiagnosticSink, name::name};
use ra_arena::map::ArenaMap;
use ra_prof::profile;
use ra_syntax::SmolStr;

use super::{
    primitive::{FloatTy, IntTy},
    traits::{Guidance, Obligation, ProjectionPredicate, Solution},
    ApplicationTy, GenericPredicate, InEnvironment, ProjectionTy, Substs, TraitEnvironment,
    TraitRef, Ty, TypeCtor, TypeWalk, Uncertain,
};
use crate::{
    db::HirDatabase, infer::diagnostics::InferenceDiagnostic, lower::ImplTraitLoweringMode,
};

pub(crate) use unify::unify;

macro_rules! ty_app {
    ($ctor:pat, $param:pat) => {
        crate::Ty::Apply(crate::ApplicationTy { ctor: $ctor, parameters: $param })
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
pub(crate) fn infer_query(db: &impl HirDatabase, def: DefWithBodyId) -> Arc<InferenceResult> {
    let _p = profile("infer_query");
    let resolver = def.resolver(db);
    let mut ctx = InferenceContext::new(db, def, resolver);

    match def {
        DefWithBodyId::ConstId(c) => ctx.collect_const(&db.const_data(c)),
        DefWithBodyId::FunctionId(f) => ctx.collect_fn(&db.function_data(f)),
        DefWithBodyId::StaticId(s) => ctx.collect_const(&db.static_data(s)),
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
    method_resolutions: FxHashMap<ExprId, FunctionId>,
    /// For each field access expr, records the field it resolves to.
    field_resolutions: FxHashMap<ExprId, StructFieldId>,
    /// For each field in record literal, records the field it resolves to.
    record_field_resolutions: FxHashMap<ExprId, StructFieldId>,
    /// For each struct literal, records the variant it resolves to.
    variant_resolutions: FxHashMap<ExprOrPatId, VariantId>,
    /// For each associated item record what it resolves to
    assoc_resolutions: FxHashMap<ExprOrPatId, AssocItemId>,
    diagnostics: Vec<InferenceDiagnostic>,
    pub type_of_expr: ArenaMap<ExprId, Ty>,
    pub type_of_pat: ArenaMap<PatId, Ty>,
    pub(super) type_mismatches: ArenaMap<ExprId, TypeMismatch>,
}

impl InferenceResult {
    pub fn method_resolution(&self, expr: ExprId) -> Option<FunctionId> {
        self.method_resolutions.get(&expr).copied()
    }
    pub fn field_resolution(&self, expr: ExprId) -> Option<StructFieldId> {
        self.field_resolutions.get(&expr).copied()
    }
    pub fn record_field_resolution(&self, expr: ExprId) -> Option<StructFieldId> {
        self.record_field_resolutions.get(&expr).copied()
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
        self.type_mismatches.get(expr)
    }
    pub fn add_diagnostics(
        &self,
        db: &impl HirDatabase,
        owner: FunctionId,
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
    owner: DefWithBodyId,
    body: Arc<Body>,
    resolver: Resolver,
    table: unify::InferenceTable,
    trait_env: Arc<TraitEnvironment>,
    obligations: Vec<Obligation>,
    result: InferenceResult,
    /// The return type of the function being inferred, or the closure if we're
    /// currently within one.
    ///
    /// We might consider using a nested inference context for checking
    /// closures, but currently this is the only field that will change there,
    /// so it doesn't make sense.
    return_ty: Ty,
}

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    fn new(db: &'a D, owner: DefWithBodyId, resolver: Resolver) -> Self {
        InferenceContext {
            result: InferenceResult::default(),
            table: unify::InferenceTable::new(),
            obligations: Vec::default(),
            return_ty: Ty::Unknown, // set in collect_fn_signature
            trait_env: TraitEnvironment::lower(db, &resolver),
            db,
            owner,
            body: db.body(owner),
            resolver,
        }
    }

    fn resolve_all(mut self) -> InferenceResult {
        // FIXME resolve obligations as well (use Guidance if necessary)
        let mut result = std::mem::take(&mut self.result);
        for ty in result.type_of_expr.values_mut() {
            let resolved = self.table.resolve_ty_completely(mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        for ty in result.type_of_pat.values_mut() {
            let resolved = self.table.resolve_ty_completely(mem::replace(ty, Ty::Unknown));
            *ty = resolved;
        }
        result
    }

    fn write_expr_ty(&mut self, expr: ExprId, ty: Ty) {
        self.result.type_of_expr.insert(expr, ty);
    }

    fn write_method_resolution(&mut self, expr: ExprId, func: FunctionId) {
        self.result.method_resolutions.insert(expr, func);
    }

    fn write_field_resolution(&mut self, expr: ExprId, field: StructFieldId) {
        self.result.field_resolutions.insert(expr, field);
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
        let ty = Ty::from_hir(&ctx, type_ref);
        let ty = self.insert_type_vars(ty);
        self.normalize_associated_types_in(ty)
    }

    fn make_ty(&mut self, type_ref: &TypeRef) -> Ty {
        self.make_ty_with_mode(type_ref, ImplTraitLoweringMode::Disallowed)
    }

    /// Replaces Ty::Unknown by a new type var, so we can maybe still infer it.
    fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty {
            Ty::Unknown => self.table.new_type_var(),
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Int(Uncertain::Unknown), .. }) => {
                self.table.new_integer_var()
            }
            Ty::Apply(ApplicationTy { ctor: TypeCtor::Float(Uncertain::Unknown), .. }) => {
                self.table.new_float_var()
            }
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

    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        self.table.unify(ty1, ty2)
    }

    /// Resolves the type as far as currently possible, replacing type variables
    /// by their known types. All types returned by the infer_* functions should
    /// be resolved as far as possible, i.e. contain no type variables with
    /// known type.
    fn resolve_ty_as_possible(&mut self, ty: Ty) -> Ty {
        self.resolve_obligations_as_possible();

        self.table.resolve_ty_as_possible(ty)
    }

    fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        self.table.resolve_ty_shallow(ty)
    }

    fn resolve_associated_type(&mut self, inner_ty: Ty, assoc_ty: Option<TypeAliasId>) -> Ty {
        self.resolve_associated_type_with_params(inner_ty, assoc_ty, &[])
    }

    fn resolve_associated_type_with_params(
        &mut self,
        inner_ty: Ty,
        assoc_ty: Option<TypeAliasId>,
        params: &[Ty],
    ) -> Ty {
        match assoc_ty {
            Some(res_assoc_ty) => {
                // FIXME:
                // Check if inner_ty is is `impl Trait` and contained input TypeAlias id
                // this is a workaround while Chalk assoc type projection doesn't always work yet,
                // but once that is fixed I don't think we should keep this
                // (we'll probably change how associated types are resolved anyway)
                if let Ty::Opaque(ref predicates) = inner_ty {
                    for p in predicates.iter() {
                        if let GenericPredicate::Projection(projection) = p {
                            if projection.projection_ty.associated_ty == res_assoc_ty {
                                if let ty_app!(_, params) = &projection.ty {
                                    if params.len() == 0 {
                                        return projection.ty.clone();
                                    }
                                }
                            }
                        }
                    }
                }

                let ty = self.table.new_type_var();
                let builder = Substs::build_for_def(self.db, res_assoc_ty)
                    .push(inner_ty)
                    .fill(params.iter().cloned());
                let projection = ProjectionPredicate {
                    ty: ty.clone(),
                    projection_ty: ProjectionTy {
                        associated_ty: res_assoc_ty,
                        parameters: builder.build(),
                    },
                };
                self.obligations.push(Obligation::Projection(projection));
                self.resolve_ty_as_possible(ty)
            }
            None => Ty::Unknown,
        }
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    fn normalize_associated_types_in(&mut self, ty: Ty) -> Ty {
        let ty = self.resolve_ty_as_possible(ty);
        ty.fold(&mut |ty| match ty {
            Ty::Projection(proj_ty) => self.normalize_projection_ty(proj_ty),
            _ => ty,
        })
    }

    fn normalize_projection_ty(&mut self, proj_ty: ProjectionTy) -> Ty {
        let var = self.table.new_type_var();
        let predicate = ProjectionPredicate { projection_ty: proj_ty, ty: var.clone() };
        let obligation = Obligation::Projection(predicate);
        self.obligations.push(obligation);
        var
    }

    fn resolve_variant(&mut self, path: Option<&Path>) -> (Ty, Option<VariantId>) {
        let path = match path {
            Some(path) => path,
            None => return (Ty::Unknown, None),
        };
        let resolver = &self.resolver;
        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver);
        // FIXME: this should resolve assoc items as well, see this example:
        // https://play.rust-lang.org/?gist=087992e9e22495446c01c0d4e2d69521
        return match resolver.resolve_path_in_type_ns_fully(self.db, path.mod_path()) {
            Some(TypeNs::AdtId(AdtId::StructId(strukt))) => {
                let substs = Ty::substs_from_path(&ctx, path, strukt.into());
                let ty = self.db.ty(strukt.into());
                let ty = self.insert_type_vars(ty.subst(&substs));
                (ty, Some(strukt.into()))
            }
            Some(TypeNs::EnumVariantId(var)) => {
                let substs = Ty::substs_from_path(&ctx, path, var.into());
                let ty = self.db.ty(var.parent.into());
                let ty = self.insert_type_vars(ty.subst(&substs));
                (ty, Some(var.into()))
            }
            Some(TypeNs::SelfType(impl_id)) => {
                let generics = crate::utils::generics(self.db, impl_id.into());
                let substs = Substs::type_params_for_generics(&generics);
                let ty = self.db.impl_self_ty(impl_id).subst(&substs);
                let variant = ty_variant(&ty);
                (ty, variant)
            }
            Some(TypeNs::TypeAliasId(it)) => {
                let substs = Substs::build_for_def(self.db, it)
                    .fill(std::iter::repeat_with(|| self.table.new_type_var()))
                    .build();
                let ty = self.db.ty(it.into()).subst(&substs);
                let variant = ty_variant(&ty);
                (ty, variant)
            }
            Some(_) | None => (Ty::Unknown, None),
        };

        fn ty_variant(ty: &Ty) -> Option<VariantId> {
            ty.as_adt().and_then(|(adt_id, _)| match adt_id {
                AdtId::StructId(s) => Some(VariantId::StructId(s)),
                AdtId::UnionId(u) => Some(VariantId::UnionId(u)),
                AdtId::EnumId(_) => {
                    // Error E0071, expected struct, variant or union type, found enum `Foo`
                    None
                }
            })
        }
    }

    fn collect_const(&mut self, data: &ConstData) {
        self.return_ty = self.make_ty(&data.type_ref);
    }

    fn collect_fn(&mut self, data: &FunctionData) {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ctx = crate::lower::TyLoweringContext::new(self.db, &self.resolver)
            .with_impl_trait_mode(ImplTraitLoweringMode::Param);
        let param_tys =
            data.params.iter().map(|type_ref| Ty::from_hir(&ctx, type_ref)).collect::<Vec<_>>();
        for (ty, pat) in param_tys.into_iter().zip(body.params.iter()) {
            let ty = self.insert_type_vars(ty);
            let ty = self.normalize_associated_types_in(ty);

            self.infer_pat(*pat, &ty, BindingMode::default());
        }
        let return_ty = self.make_ty_with_mode(&data.ret_type, ImplTraitLoweringMode::Disallowed); // FIXME implement RPIT
        self.return_ty = return_ty;
    }

    fn infer_body(&mut self) {
        self.infer_expr_coerce(self.body.body_expr, &Expectation::has_type(self.return_ty.clone()));
    }

    fn resolve_lang_item(&self, name: &str) -> Option<LangItemTarget> {
        let krate = self.resolver.krate()?;
        let name = SmolStr::new_inline_from_ascii(name.len(), name.as_bytes());
        self.db.lang_item(krate, name)
    }

    fn resolve_into_iter_item(&self) -> Option<TypeAliasId> {
        let path = path![std::iter::IntoIterator];
        let trait_ = self.resolver.resolve_known_trait(self.db, &path)?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Item])
    }

    fn resolve_ops_try_ok(&self) -> Option<TypeAliasId> {
        let path = path![std::ops::Try];
        let trait_ = self.resolver.resolve_known_trait(self.db, &path)?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Ok])
    }

    fn resolve_ops_neg_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_lang_item("neg")?.as_trait()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }

    fn resolve_ops_not_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_lang_item("not")?.as_trait()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }

    fn resolve_future_future_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_lang_item("future_trait")?.as_trait()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }

    fn resolve_boxed_box(&self) -> Option<AdtId> {
        let struct_ = self.resolve_lang_item("owned_box")?.as_struct()?;
        Some(struct_.into())
    }

    fn resolve_range_full(&self) -> Option<AdtId> {
        let path = path![std::ops::RangeFull];
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(struct_.into())
    }

    fn resolve_range(&self) -> Option<AdtId> {
        let path = path![std::ops::Range];
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(struct_.into())
    }

    fn resolve_range_inclusive(&self) -> Option<AdtId> {
        let path = path![std::ops::RangeInclusive];
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(struct_.into())
    }

    fn resolve_range_from(&self) -> Option<AdtId> {
        let path = path![std::ops::RangeFrom];
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(struct_.into())
    }

    fn resolve_range_to(&self) -> Option<AdtId> {
        let path = path![std::ops::RangeTo];
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(struct_.into())
    }

    fn resolve_range_to_inclusive(&self) -> Option<AdtId> {
        let path = path![std::ops::RangeToInclusive];
        let struct_ = self.resolver.resolve_known_struct(self.db, &path)?;
        Some(struct_.into())
    }

    fn resolve_ops_index(&self) -> Option<TraitId> {
        self.resolve_lang_item("index")?.as_trait()
    }

    fn resolve_ops_index_output(&self) -> Option<TypeAliasId> {
        let trait_ = self.resolve_ops_index()?;
        self.db.trait_data(trait_).associated_type_by_name(&name![Output])
    }
}

/// The kinds of placeholders we need during type inference. There's separate
/// values for general types, and for integer and float variables. The latter
/// two are used for inference of literal values (e.g. `100` could be one of
/// several integer types).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InferTy {
    TypeVar(unify::TypeVarId),
    IntVar(unify::TypeVarId),
    FloatVar(unify::TypeVarId),
    MaybeNeverTypeVar(unify::TypeVarId),
}

impl InferTy {
    fn to_inner(self) -> unify::TypeVarId {
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
            InferTy::IntVar(..) => Ty::simple(TypeCtor::Int(Uncertain::Known(IntTy::i32()))),
            InferTy::FloatVar(..) => Ty::simple(TypeCtor::Float(Uncertain::Known(FloatTy::f64()))),
            InferTy::MaybeNeverTypeVar(..) => Ty::simple(TypeCtor::Never),
        }
    }
}

/// When inferring an expression, we propagate downward whatever type hint we
/// are able in the form of an `Expectation`.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Expectation {
    ty: Ty,
    /// See the `rvalue_hint` method.
    rvalue_hint: bool,
}

impl Expectation {
    /// The expectation that the type of the expression needs to equal the given
    /// type.
    fn has_type(ty: Ty) -> Self {
        Expectation { ty, rvalue_hint: false }
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
    fn rvalue_hint(ty: Ty) -> Self {
        Expectation { ty, rvalue_hint: true }
    }

    /// This expresses no expectation on the type.
    fn none() -> Self {
        Expectation { ty: Ty::Unknown, rvalue_hint: false }
    }

    fn coercion_target(&self) -> &Ty {
        if self.rvalue_hint {
            &Ty::Unknown
        } else {
            &self.ty
        }
    }
}

mod diagnostics {
    use hir_def::{expr::ExprId, src::HasSource, FunctionId, Lookup};
    use hir_expand::diagnostics::DiagnosticSink;

    use crate::{db::HirDatabase, diagnostics::NoSuchField};

    #[derive(Debug, PartialEq, Eq, Clone)]
    pub(super) enum InferenceDiagnostic {
        NoSuchField { expr: ExprId, field: usize },
    }

    impl InferenceDiagnostic {
        pub(super) fn add_to(
            &self,
            db: &impl HirDatabase,
            owner: FunctionId,
            sink: &mut DiagnosticSink,
        ) {
            match self {
                InferenceDiagnostic::NoSuchField { expr, field } => {
                    let file = owner.lookup(db).source(db).file_id;
                    let (_, source_map) = db.body_with_source_map(owner.into());
                    let field = source_map.field_syntax(*expr, *field);
                    sink.push(NoSuchField { file, field })
                }
            }
        }
    }
}
