//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `Ty::from_hir`.
//!  - Building the type for an item: This happens through the `type_for_def` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
use std::iter;
use std::sync::Arc;

use hir_def::{
    adt::StructKind,
    builtin_type::BuiltinType,
    generics::{TypeParamProvenance, WherePredicate, WherePredicateTarget},
    path::{GenericArg, Path, PathSegment, PathSegments},
    resolver::{HasResolver, Resolver, TypeNs},
    type_ref::{TypeBound, TypeRef},
    AdtId, AssocContainerId, ConstId, EnumId, EnumVariantId, FunctionId, GenericDefId, HasModule,
    ImplId, LocalStructFieldId, Lookup, StaticId, StructId, TraitId, TypeAliasId, TypeParamId,
    UnionId, VariantId,
};
use ra_arena::map::ArenaMap;
use ra_db::CrateId;

use crate::{
    db::HirDatabase,
    primitive::{FloatTy, IntTy},
    utils::{
        all_super_traits, associated_type_by_name_including_super_traits, generics, make_mut_slice,
        variant_data,
    },
    Binders, FnSig, GenericPredicate, PolyFnSig, ProjectionPredicate, ProjectionTy, Substs,
    TraitEnvironment, TraitRef, Ty, TypeCtor,
};

#[derive(Debug)]
pub struct TyLoweringContext<'a> {
    pub db: &'a dyn HirDatabase,
    pub resolver: &'a Resolver,
    /// Note: Conceptually, it's thinkable that we could be in a location where
    /// some type params should be represented as placeholders, and others
    /// should be converted to variables. I think in practice, this isn't
    /// possible currently, so this should be fine for now.
    pub type_param_mode: TypeParamLoweringMode,
    pub impl_trait_mode: ImplTraitLoweringMode,
    pub impl_trait_counter: std::cell::Cell<u16>,
}

impl<'a> TyLoweringContext<'a> {
    pub fn new(db: &'a dyn HirDatabase, resolver: &'a Resolver) -> Self {
        let impl_trait_counter = std::cell::Cell::new(0);
        let impl_trait_mode = ImplTraitLoweringMode::Disallowed;
        let type_param_mode = TypeParamLoweringMode::Placeholder;
        Self { db, resolver, impl_trait_mode, impl_trait_counter, type_param_mode }
    }

    pub fn with_impl_trait_mode(self, impl_trait_mode: ImplTraitLoweringMode) -> Self {
        Self { impl_trait_mode, ..self }
    }

    pub fn with_type_param_mode(self, type_param_mode: TypeParamLoweringMode) -> Self {
        Self { type_param_mode, ..self }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImplTraitLoweringMode {
    /// `impl Trait` gets lowered into an opaque type that doesn't unify with
    /// anything except itself. This is used in places where values flow 'out',
    /// i.e. for arguments of the function we're currently checking, and return
    /// types of functions we're calling.
    Opaque,
    /// `impl Trait` gets lowered into a type variable. Used for argument
    /// position impl Trait when inside the respective function, since it allows
    /// us to support that without Chalk.
    Param,
    /// `impl Trait` gets lowered into a variable that can unify with some
    /// type. This is used in places where values flow 'in', i.e. for arguments
    /// of functions we're calling, and the return type of the function we're
    /// currently checking.
    Variable,
    /// `impl Trait` is disallowed and will be an error.
    Disallowed,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TypeParamLoweringMode {
    Placeholder,
    Variable,
}

impl Ty {
    pub fn from_hir(ctx: &TyLoweringContext<'_>, type_ref: &TypeRef) -> Self {
        Ty::from_hir_ext(ctx, type_ref).0
    }
    pub fn from_hir_ext(ctx: &TyLoweringContext<'_>, type_ref: &TypeRef) -> (Self, Option<TypeNs>) {
        let mut res = None;
        let ty = match type_ref {
            TypeRef::Never => Ty::simple(TypeCtor::Never),
            TypeRef::Tuple(inner) => {
                let inner_tys: Arc<[Ty]> = inner.iter().map(|tr| Ty::from_hir(ctx, tr)).collect();
                Ty::apply(
                    TypeCtor::Tuple { cardinality: inner_tys.len() as u16 },
                    Substs(inner_tys),
                )
            }
            TypeRef::Path(path) => {
                let (ty, res_) = Ty::from_hir_path(ctx, path);
                res = res_;
                ty
            }
            TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::apply_one(TypeCtor::RawPtr(*mutability), inner_ty)
            }
            TypeRef::Array(inner) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::apply_one(TypeCtor::Array, inner_ty)
            }
            TypeRef::Slice(inner) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::apply_one(TypeCtor::Slice, inner_ty)
            }
            TypeRef::Reference(inner, mutability) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::apply_one(TypeCtor::Ref(*mutability), inner_ty)
            }
            TypeRef::Placeholder => Ty::Unknown,
            TypeRef::Fn(params) => {
                let sig = Substs(params.iter().map(|tr| Ty::from_hir(ctx, tr)).collect());
                Ty::apply(TypeCtor::FnPtr { num_args: sig.len() as u16 - 1 }, sig)
            }
            TypeRef::DynTrait(bounds) => {
                let self_ty = Ty::Bound(0);
                let predicates = bounds
                    .iter()
                    .flat_map(|b| GenericPredicate::from_type_bound(ctx, b, self_ty.clone()))
                    .collect();
                Ty::Dyn(predicates)
            }
            TypeRef::ImplTrait(bounds) => {
                match ctx.impl_trait_mode {
                    ImplTraitLoweringMode::Opaque => {
                        let self_ty = Ty::Bound(0);
                        let predicates = bounds
                            .iter()
                            .flat_map(|b| {
                                GenericPredicate::from_type_bound(ctx, b, self_ty.clone())
                            })
                            .collect();
                        Ty::Opaque(predicates)
                    }
                    ImplTraitLoweringMode::Param => {
                        let idx = ctx.impl_trait_counter.get();
                        ctx.impl_trait_counter.set(idx + 1);
                        if let Some(def) = ctx.resolver.generic_def() {
                            let generics = generics(ctx.db.upcast(), def);
                            let param = generics
                                .iter()
                                .filter(|(_, data)| {
                                    data.provenance == TypeParamProvenance::ArgumentImplTrait
                                })
                                .nth(idx as usize)
                                .map_or(Ty::Unknown, |(id, _)| Ty::Placeholder(id));
                            param
                        } else {
                            Ty::Unknown
                        }
                    }
                    ImplTraitLoweringMode::Variable => {
                        let idx = ctx.impl_trait_counter.get();
                        ctx.impl_trait_counter.set(idx + 1);
                        let (parent_params, self_params, list_params, _impl_trait_params) =
                            if let Some(def) = ctx.resolver.generic_def() {
                                let generics = generics(ctx.db.upcast(), def);
                                generics.provenance_split()
                            } else {
                                (0, 0, 0, 0)
                            };
                        Ty::Bound(
                            idx as u32
                                + parent_params as u32
                                + self_params as u32
                                + list_params as u32,
                        )
                    }
                    ImplTraitLoweringMode::Disallowed => {
                        // FIXME: report error
                        Ty::Unknown
                    }
                }
            }
            TypeRef::Error => Ty::Unknown,
        };
        (ty, res)
    }

    /// This is only for `generic_predicates_for_param`, where we can't just
    /// lower the self types of the predicates since that could lead to cycles.
    /// So we just check here if the `type_ref` resolves to a generic param, and which.
    fn from_hir_only_param(ctx: &TyLoweringContext<'_>, type_ref: &TypeRef) -> Option<TypeParamId> {
        let path = match type_ref {
            TypeRef::Path(path) => path,
            _ => return None,
        };
        if path.type_anchor().is_some() {
            return None;
        }
        if path.segments().len() > 1 {
            return None;
        }
        let resolution =
            match ctx.resolver.resolve_path_in_type_ns(ctx.db.upcast(), path.mod_path()) {
                Some((it, None)) => it,
                _ => return None,
            };
        if let TypeNs::GenericParam(param_id) = resolution {
            Some(param_id)
        } else {
            None
        }
    }

    pub(crate) fn from_type_relative_path(
        ctx: &TyLoweringContext<'_>,
        ty: Ty,
        // We need the original resolution to lower `Self::AssocTy` correctly
        res: Option<TypeNs>,
        remaining_segments: PathSegments<'_>,
    ) -> (Ty, Option<TypeNs>) {
        if remaining_segments.len() == 1 {
            // resolve unselected assoc types
            let segment = remaining_segments.first().unwrap();
            (Ty::select_associated_type(ctx, ty, res, segment), None)
        } else if remaining_segments.len() > 1 {
            // FIXME report error (ambiguous associated type)
            (Ty::Unknown, None)
        } else {
            (ty, res)
        }
    }

    pub(crate) fn from_partly_resolved_hir_path(
        ctx: &TyLoweringContext<'_>,
        resolution: TypeNs,
        resolved_segment: PathSegment<'_>,
        remaining_segments: PathSegments<'_>,
    ) -> (Ty, Option<TypeNs>) {
        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                // if this is a bare dyn Trait, we'll directly put the required ^0 for the self type in there
                let self_ty = if remaining_segments.len() == 0 { Some(Ty::Bound(0)) } else { None };
                let trait_ref =
                    TraitRef::from_resolved_path(ctx, trait_, resolved_segment, self_ty);
                let ty = if remaining_segments.len() == 1 {
                    let segment = remaining_segments.first().unwrap();
                    let associated_ty = associated_type_by_name_including_super_traits(
                        ctx.db.upcast(),
                        trait_ref.trait_,
                        &segment.name,
                    );
                    match associated_ty {
                        Some(associated_ty) => {
                            // FIXME handle type parameters on the segment
                            Ty::Projection(ProjectionTy {
                                associated_ty,
                                parameters: trait_ref.substs,
                            })
                        }
                        None => {
                            // FIXME: report error (associated type not found)
                            Ty::Unknown
                        }
                    }
                } else if remaining_segments.len() > 1 {
                    // FIXME report error (ambiguous associated type)
                    Ty::Unknown
                } else {
                    Ty::Dyn(Arc::new([GenericPredicate::Implemented(trait_ref)]))
                };
                return (ty, None);
            }
            TypeNs::GenericParam(param_id) => {
                let generics = generics(
                    ctx.db.upcast(),
                    ctx.resolver.generic_def().expect("generics in scope"),
                );
                match ctx.type_param_mode {
                    TypeParamLoweringMode::Placeholder => Ty::Placeholder(param_id),
                    TypeParamLoweringMode::Variable => {
                        let idx = generics.param_idx(param_id).expect("matching generics");
                        Ty::Bound(idx)
                    }
                }
            }
            TypeNs::SelfType(impl_id) => {
                let generics = generics(ctx.db.upcast(), impl_id.into());
                let substs = match ctx.type_param_mode {
                    TypeParamLoweringMode::Placeholder => {
                        Substs::type_params_for_generics(&generics)
                    }
                    TypeParamLoweringMode::Variable => Substs::bound_vars(&generics),
                };
                ctx.db.impl_self_ty(impl_id).subst(&substs)
            }
            TypeNs::AdtSelfType(adt) => {
                let generics = generics(ctx.db.upcast(), adt.into());
                let substs = match ctx.type_param_mode {
                    TypeParamLoweringMode::Placeholder => {
                        Substs::type_params_for_generics(&generics)
                    }
                    TypeParamLoweringMode::Variable => Substs::bound_vars(&generics),
                };
                ctx.db.ty(adt.into()).subst(&substs)
            }

            TypeNs::AdtId(it) => Ty::from_hir_path_inner(ctx, resolved_segment, it.into()),
            TypeNs::BuiltinType(it) => Ty::from_hir_path_inner(ctx, resolved_segment, it.into()),
            TypeNs::TypeAliasId(it) => Ty::from_hir_path_inner(ctx, resolved_segment, it.into()),
            // FIXME: report error
            TypeNs::EnumVariantId(_) => return (Ty::Unknown, None),
        };

        Ty::from_type_relative_path(ctx, ty, Some(resolution), remaining_segments)
    }

    pub(crate) fn from_hir_path(ctx: &TyLoweringContext<'_>, path: &Path) -> (Ty, Option<TypeNs>) {
        // Resolve the path (in type namespace)
        if let Some(type_ref) = path.type_anchor() {
            let (ty, res) = Ty::from_hir_ext(ctx, &type_ref);
            return Ty::from_type_relative_path(ctx, ty, res, path.segments());
        }
        let (resolution, remaining_index) =
            match ctx.resolver.resolve_path_in_type_ns(ctx.db.upcast(), path.mod_path()) {
                Some(it) => it,
                None => return (Ty::Unknown, None),
            };
        let (resolved_segment, remaining_segments) = match remaining_index {
            None => (
                path.segments().last().expect("resolved path has at least one element"),
                PathSegments::EMPTY,
            ),
            Some(i) => (path.segments().get(i - 1).unwrap(), path.segments().skip(i)),
        };
        Ty::from_partly_resolved_hir_path(ctx, resolution, resolved_segment, remaining_segments)
    }

    fn select_associated_type(
        ctx: &TyLoweringContext<'_>,
        self_ty: Ty,
        res: Option<TypeNs>,
        segment: PathSegment<'_>,
    ) -> Ty {
        let traits_from_env: Vec<_> = match res {
            Some(TypeNs::SelfType(impl_id)) => match ctx.db.impl_trait(impl_id) {
                None => return Ty::Unknown,
                Some(trait_ref) => vec![trait_ref.value.trait_],
            },
            Some(TypeNs::GenericParam(param_id)) => {
                let predicates = ctx.db.generic_predicates_for_param(param_id);
                predicates
                    .iter()
                    .filter_map(|pred| match &pred.value {
                        GenericPredicate::Implemented(tr) => Some(tr.trait_),
                        _ => None,
                    })
                    .collect()
            }
            _ => return Ty::Unknown,
        };
        let traits = traits_from_env.into_iter().flat_map(|t| all_super_traits(ctx.db.upcast(), t));
        for t in traits {
            if let Some(associated_ty) = ctx.db.trait_data(t).associated_type_by_name(&segment.name)
            {
                let substs =
                    Substs::build_for_def(ctx.db, t).push(self_ty).fill_with_unknown().build();
                // FIXME handle type parameters on the segment
                return Ty::Projection(ProjectionTy { associated_ty, parameters: substs });
            }
        }
        Ty::Unknown
    }

    fn from_hir_path_inner(
        ctx: &TyLoweringContext<'_>,
        segment: PathSegment<'_>,
        typable: TyDefId,
    ) -> Ty {
        let generic_def = match typable {
            TyDefId::BuiltinType(_) => None,
            TyDefId::AdtId(it) => Some(it.into()),
            TyDefId::TypeAliasId(it) => Some(it.into()),
        };
        let substs = substs_from_path_segment(ctx, segment, generic_def, false);
        ctx.db.ty(typable).subst(&substs)
    }

    /// Collect generic arguments from a path into a `Substs`. See also
    /// `create_substs_for_ast_path` and `def_to_ty` in rustc.
    pub(super) fn substs_from_path(
        ctx: &TyLoweringContext<'_>,
        path: &Path,
        // Note that we don't call `db.value_type(resolved)` here,
        // `ValueTyDefId` is just a convenient way to pass generics and
        // special-case enum variants
        resolved: ValueTyDefId,
    ) -> Substs {
        let last = path.segments().last().expect("path should have at least one segment");
        let (segment, generic_def) = match resolved {
            ValueTyDefId::FunctionId(it) => (last, Some(it.into())),
            ValueTyDefId::StructId(it) => (last, Some(it.into())),
            ValueTyDefId::ConstId(it) => (last, Some(it.into())),
            ValueTyDefId::StaticId(_) => (last, None),
            ValueTyDefId::EnumVariantId(var) => {
                // the generic args for an enum variant may be either specified
                // on the segment referring to the enum, or on the segment
                // referring to the variant. So `Option::<T>::None` and
                // `Option::None::<T>` are both allowed (though the former is
                // preferred). See also `def_ids_for_path_segments` in rustc.
                let len = path.segments().len();
                let penultimate = if len >= 2 { path.segments().get(len - 2) } else { None };
                let segment = match penultimate {
                    Some(segment) if segment.args_and_bindings.is_some() => segment,
                    _ => last,
                };
                (segment, Some(var.parent.into()))
            }
        };
        substs_from_path_segment(ctx, segment, generic_def, false)
    }
}

pub(super) fn substs_from_path_segment(
    ctx: &TyLoweringContext<'_>,
    segment: PathSegment<'_>,
    def_generic: Option<GenericDefId>,
    _add_self_param: bool,
) -> Substs {
    let mut substs = Vec::new();
    let def_generics = def_generic.map(|def| generics(ctx.db.upcast(), def));

    let (parent_params, self_params, type_params, impl_trait_params) =
        def_generics.map_or((0, 0, 0, 0), |g| g.provenance_split());
    substs.extend(iter::repeat(Ty::Unknown).take(parent_params));
    if let Some(generic_args) = &segment.args_and_bindings {
        if !generic_args.has_self_type {
            substs.extend(iter::repeat(Ty::Unknown).take(self_params));
        }
        let expected_num =
            if generic_args.has_self_type { self_params + type_params } else { type_params };
        let skip = if generic_args.has_self_type && self_params == 0 { 1 } else { 0 };
        // if args are provided, it should be all of them, but we can't rely on that
        for arg in generic_args.args.iter().skip(skip).take(expected_num) {
            match arg {
                GenericArg::Type(type_ref) => {
                    let ty = Ty::from_hir(ctx, type_ref);
                    substs.push(ty);
                }
            }
        }
    }
    let total_len = parent_params + self_params + type_params + impl_trait_params;
    // add placeholders for args that were not provided
    for _ in substs.len()..total_len {
        substs.push(Ty::Unknown);
    }
    assert_eq!(substs.len(), total_len);

    // handle defaults
    if let Some(def_generic) = def_generic {
        let default_substs = ctx.db.generic_defaults(def_generic);
        assert_eq!(substs.len(), default_substs.len());

        for (i, default_ty) in default_substs.iter().enumerate() {
            if substs[i] == Ty::Unknown {
                substs[i] = default_ty.clone();
            }
        }
    }

    Substs(substs.into())
}

impl TraitRef {
    fn from_path(
        ctx: &TyLoweringContext<'_>,
        path: &Path,
        explicit_self_ty: Option<Ty>,
    ) -> Option<Self> {
        let resolved =
            match ctx.resolver.resolve_path_in_type_ns_fully(ctx.db.upcast(), path.mod_path())? {
                TypeNs::TraitId(tr) => tr,
                _ => return None,
            };
        let segment = path.segments().last().expect("path should have at least one segment");
        Some(TraitRef::from_resolved_path(ctx, resolved, segment, explicit_self_ty))
    }

    pub(crate) fn from_resolved_path(
        ctx: &TyLoweringContext<'_>,
        resolved: TraitId,
        segment: PathSegment<'_>,
        explicit_self_ty: Option<Ty>,
    ) -> Self {
        let mut substs = TraitRef::substs_from_path(ctx, segment, resolved);
        if let Some(self_ty) = explicit_self_ty {
            make_mut_slice(&mut substs.0)[0] = self_ty;
        }
        TraitRef { trait_: resolved, substs }
    }

    fn from_hir(
        ctx: &TyLoweringContext<'_>,
        type_ref: &TypeRef,
        explicit_self_ty: Option<Ty>,
    ) -> Option<Self> {
        let path = match type_ref {
            TypeRef::Path(path) => path,
            _ => return None,
        };
        TraitRef::from_path(ctx, path, explicit_self_ty)
    }

    fn substs_from_path(
        ctx: &TyLoweringContext<'_>,
        segment: PathSegment<'_>,
        resolved: TraitId,
    ) -> Substs {
        let has_self_param =
            segment.args_and_bindings.as_ref().map(|a| a.has_self_type).unwrap_or(false);
        substs_from_path_segment(ctx, segment, Some(resolved.into()), !has_self_param)
    }

    pub(crate) fn from_type_bound(
        ctx: &TyLoweringContext<'_>,
        bound: &TypeBound,
        self_ty: Ty,
    ) -> Option<TraitRef> {
        match bound {
            TypeBound::Path(path) => TraitRef::from_path(ctx, path, Some(self_ty)),
            TypeBound::Error => None,
        }
    }
}

impl GenericPredicate {
    pub(crate) fn from_where_predicate<'a>(
        ctx: &'a TyLoweringContext<'a>,
        where_predicate: &'a WherePredicate,
    ) -> impl Iterator<Item = GenericPredicate> + 'a {
        let self_ty = match &where_predicate.target {
            WherePredicateTarget::TypeRef(type_ref) => Ty::from_hir(ctx, type_ref),
            WherePredicateTarget::TypeParam(param_id) => {
                let generic_def = ctx.resolver.generic_def().expect("generics in scope");
                let generics = generics(ctx.db.upcast(), generic_def);
                let param_id = hir_def::TypeParamId { parent: generic_def, local_id: *param_id };
                match ctx.type_param_mode {
                    TypeParamLoweringMode::Placeholder => Ty::Placeholder(param_id),
                    TypeParamLoweringMode::Variable => {
                        let idx = generics.param_idx(param_id).expect("matching generics");
                        Ty::Bound(idx)
                    }
                }
            }
        };
        GenericPredicate::from_type_bound(ctx, &where_predicate.bound, self_ty)
    }

    pub(crate) fn from_type_bound<'a>(
        ctx: &'a TyLoweringContext<'a>,
        bound: &'a TypeBound,
        self_ty: Ty,
    ) -> impl Iterator<Item = GenericPredicate> + 'a {
        let trait_ref = TraitRef::from_type_bound(ctx, bound, self_ty);
        iter::once(trait_ref.clone().map_or(GenericPredicate::Error, GenericPredicate::Implemented))
            .chain(
                trait_ref
                    .into_iter()
                    .flat_map(move |tr| assoc_type_bindings_from_type_bound(ctx, bound, tr)),
            )
    }
}

fn assoc_type_bindings_from_type_bound<'a>(
    ctx: &'a TyLoweringContext<'a>,
    bound: &'a TypeBound,
    trait_ref: TraitRef,
) -> impl Iterator<Item = GenericPredicate> + 'a {
    let last_segment = match bound {
        TypeBound::Path(path) => path.segments().last(),
        TypeBound::Error => None,
    };
    last_segment
        .into_iter()
        .flat_map(|segment| segment.args_and_bindings.into_iter())
        .flat_map(|args_and_bindings| args_and_bindings.bindings.iter())
        .map(move |(name, type_ref)| {
            let associated_ty = associated_type_by_name_including_super_traits(
                ctx.db.upcast(),
                trait_ref.trait_,
                &name,
            );
            let associated_ty = match associated_ty {
                None => return GenericPredicate::Error,
                Some(t) => t,
            };
            let projection_ty =
                ProjectionTy { associated_ty, parameters: trait_ref.substs.clone() };
            let ty = Ty::from_hir(ctx, type_ref);
            let projection_predicate = ProjectionPredicate { projection_ty, ty };
            GenericPredicate::Projection(projection_predicate)
        })
}

/// Build the signature of a callable item (function, struct or enum variant).
pub fn callable_item_sig(db: &dyn HirDatabase, def: CallableDef) -> PolyFnSig {
    match def {
        CallableDef::FunctionId(f) => fn_sig_for_fn(db, f),
        CallableDef::StructId(s) => fn_sig_for_struct_constructor(db, s),
        CallableDef::EnumVariantId(e) => fn_sig_for_enum_variant_constructor(db, e),
    }
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalStructFieldId, Binders<Ty>>> {
    let var_data = variant_data(db.upcast(), variant_id);
    let (resolver, def): (_, GenericDefId) = match variant_id {
        VariantId::StructId(it) => (it.resolver(db.upcast()), it.into()),
        VariantId::UnionId(it) => (it.resolver(db.upcast()), it.into()),
        VariantId::EnumVariantId(it) => (it.parent.resolver(db.upcast()), it.parent.into()),
    };
    let generics = generics(db.upcast(), def);
    let mut res = ArenaMap::default();
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, Binders::new(generics.len(), Ty::from_hir(&ctx, &field_data.type_ref)))
    }
    Arc::new(res)
}

/// This query exists only to be used when resolving short-hand associated types
/// like `T::Item`.
///
/// See the analogous query in rustc and its comment:
/// https://github.com/rust-lang/rust/blob/9150f844e2624eb013ec78ca08c1d416e6644026/src/librustc_typeck/astconv.rs#L46
/// This is a query mostly to handle cycles somewhat gracefully; e.g. the
/// following bounds are disallowed: `T: Foo<U::Item>, U: Foo<T::Item>`, but
/// these are fine: `T: Foo<U::Item>, U: Foo<()>`.
pub(crate) fn generic_predicates_for_param_query(
    db: &dyn HirDatabase,
    param_id: TypeParamId,
) -> Arc<[Binders<GenericPredicate>]> {
    let resolver = param_id.parent.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let generics = generics(db.upcast(), param_id.parent);
    resolver
        .where_predicates_in_scope()
        // we have to filter out all other predicates *first*, before attempting to lower them
        .filter(|pred| match &pred.target {
            WherePredicateTarget::TypeRef(type_ref) => {
                Ty::from_hir_only_param(&ctx, type_ref) == Some(param_id)
            }
            WherePredicateTarget::TypeParam(local_id) => *local_id == param_id.local_id,
        })
        .flat_map(|pred| {
            GenericPredicate::from_where_predicate(&ctx, pred)
                .map(|p| Binders::new(generics.len(), p))
        })
        .collect()
}

pub(crate) fn generic_predicates_for_param_recover(
    _db: &dyn HirDatabase,
    _cycle: &[String],
    _param_id: &TypeParamId,
) -> Arc<[Binders<GenericPredicate>]> {
    Arc::new([])
}

impl TraitEnvironment {
    pub fn lower(db: &dyn HirDatabase, resolver: &Resolver) -> Arc<TraitEnvironment> {
        let ctx = TyLoweringContext::new(db, &resolver)
            .with_type_param_mode(TypeParamLoweringMode::Placeholder);
        let mut predicates = resolver
            .where_predicates_in_scope()
            .flat_map(|pred| GenericPredicate::from_where_predicate(&ctx, pred))
            .collect::<Vec<_>>();

        if let Some(def) = resolver.generic_def() {
            let container: Option<AssocContainerId> = match def {
                // FIXME: is there a function for this?
                GenericDefId::FunctionId(f) => Some(f.lookup(db.upcast()).container),
                GenericDefId::AdtId(_) => None,
                GenericDefId::TraitId(_) => None,
                GenericDefId::TypeAliasId(t) => Some(t.lookup(db.upcast()).container),
                GenericDefId::ImplId(_) => None,
                GenericDefId::EnumVariantId(_) => None,
                GenericDefId::ConstId(c) => Some(c.lookup(db.upcast()).container),
            };
            if let Some(AssocContainerId::TraitId(trait_id)) = container {
                // add `Self: Trait<T1, T2, ...>` to the environment in trait
                // function default implementations (and hypothetical code
                // inside consts or type aliases)
                test_utils::tested_by!(trait_self_implements_self);
                let substs = Substs::type_params(db, trait_id);
                let trait_ref = TraitRef { trait_: trait_id, substs };
                let pred = GenericPredicate::Implemented(trait_ref);

                predicates.push(pred);
            }
        }

        Arc::new(TraitEnvironment { predicates })
    }
}

/// Resolve the where clause(s) of an item with generics.
pub(crate) fn generic_predicates_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Arc<[Binders<GenericPredicate>]> {
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let generics = generics(db.upcast(), def);
    resolver
        .where_predicates_in_scope()
        .flat_map(|pred| {
            GenericPredicate::from_where_predicate(&ctx, pred)
                .map(|p| Binders::new(generics.len(), p))
        })
        .collect()
}

/// Resolve the default type params from generics
pub(crate) fn generic_defaults_query(db: &dyn HirDatabase, def: GenericDefId) -> Substs {
    let resolver = def.resolver(db.upcast());
    let ctx = TyLoweringContext::new(db, &resolver);
    let generic_params = generics(db.upcast(), def);

    let defaults = generic_params
        .iter()
        .map(|(_idx, p)| p.default.as_ref().map_or(Ty::Unknown, |t| Ty::from_hir(&ctx, t)))
        .collect();

    Substs(defaults)
}

fn fn_sig_for_fn(db: &dyn HirDatabase, def: FunctionId) -> PolyFnSig {
    let data = db.function_data(def);
    let resolver = def.resolver(db.upcast());
    let ctx_params = TyLoweringContext::new(db, &resolver)
        .with_impl_trait_mode(ImplTraitLoweringMode::Variable)
        .with_type_param_mode(TypeParamLoweringMode::Variable);
    let params = data.params.iter().map(|tr| Ty::from_hir(&ctx_params, tr)).collect::<Vec<_>>();
    let ctx_ret = ctx_params.with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
    let ret = Ty::from_hir(&ctx_ret, &data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let num_binders = generics.len();
    Binders::new(num_binders, FnSig::from_params_and_return(params, ret))
}

/// Build the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn(db: &dyn HirDatabase, def: FunctionId) -> Binders<Ty> {
    let generics = generics(db.upcast(), def.into());
    let substs = Substs::bound_vars(&generics);
    Binders::new(substs.len(), Ty::apply(TypeCtor::FnDef(def.into()), substs))
}

/// Build the declared type of a const.
fn type_for_const(db: &dyn HirDatabase, def: ConstId) -> Binders<Ty> {
    let data = db.const_data(def);
    let generics = generics(db.upcast(), def.into());
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);

    Binders::new(generics.len(), Ty::from_hir(&ctx, &data.type_ref))
}

/// Build the declared type of a static.
fn type_for_static(db: &dyn HirDatabase, def: StaticId) -> Binders<Ty> {
    let data = db.static_data(def);
    let resolver = def.resolver(db.upcast());
    let ctx = TyLoweringContext::new(db, &resolver);

    Binders::new(0, Ty::from_hir(&ctx, &data.type_ref))
}

/// Build the declared type of a static.
fn type_for_builtin(def: BuiltinType) -> Ty {
    Ty::simple(match def {
        BuiltinType::Char => TypeCtor::Char,
        BuiltinType::Bool => TypeCtor::Bool,
        BuiltinType::Str => TypeCtor::Str,
        BuiltinType::Int(t) => TypeCtor::Int(IntTy::from(t).into()),
        BuiltinType::Float(t) => TypeCtor::Float(FloatTy::from(t).into()),
    })
}

fn fn_sig_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> PolyFnSig {
    let struct_data = db.struct_data(def);
    let fields = struct_data.variant_data.fields();
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let params =
        fields.iter().map(|(_, field)| Ty::from_hir(&ctx, &field.type_ref)).collect::<Vec<_>>();
    let ret = type_for_adt(db, def.into());
    Binders::new(ret.num_binders, FnSig::from_params_and_return(params, ret.value))
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> Binders<Ty> {
    let struct_data = db.struct_data(def);
    if let StructKind::Unit = struct_data.variant_data.kind() {
        return type_for_adt(db, def.into());
    }
    let generics = generics(db.upcast(), def.into());
    let substs = Substs::bound_vars(&generics);
    Binders::new(substs.len(), Ty::apply(TypeCtor::FnDef(def.into()), substs))
}

fn fn_sig_for_enum_variant_constructor(db: &dyn HirDatabase, def: EnumVariantId) -> PolyFnSig {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id];
    let fields = var_data.variant_data.fields();
    let resolver = def.parent.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let params =
        fields.iter().map(|(_, field)| Ty::from_hir(&ctx, &field.type_ref)).collect::<Vec<_>>();
    let ret = type_for_adt(db, def.parent.into());
    Binders::new(ret.num_binders, FnSig::from_params_and_return(params, ret.value))
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor(db: &dyn HirDatabase, def: EnumVariantId) -> Binders<Ty> {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id].variant_data;
    if let StructKind::Unit = var_data.kind() {
        return type_for_adt(db, def.parent.into());
    }
    let generics = generics(db.upcast(), def.parent.into());
    let substs = Substs::bound_vars(&generics);
    Binders::new(substs.len(), Ty::apply(TypeCtor::FnDef(def.into()), substs))
}

fn type_for_adt(db: &dyn HirDatabase, adt: AdtId) -> Binders<Ty> {
    let generics = generics(db.upcast(), adt.into());
    let substs = Substs::bound_vars(&generics);
    Binders::new(substs.len(), Ty::apply(TypeCtor::Adt(adt), substs))
}

fn type_for_type_alias(db: &dyn HirDatabase, t: TypeAliasId) -> Binders<Ty> {
    let generics = generics(db.upcast(), t.into());
    let resolver = t.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let type_ref = &db.type_alias_data(t).type_ref;
    let substs = Substs::bound_vars(&generics);
    let inner = Ty::from_hir(&ctx, type_ref.as_ref().unwrap_or(&TypeRef::Error));
    Binders::new(substs.len(), inner)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableDef {
    FunctionId(FunctionId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
}
impl_froms!(CallableDef: FunctionId, StructId, EnumVariantId);

impl CallableDef {
    pub fn krate(self, db: &dyn HirDatabase) -> CrateId {
        let db = db.upcast();
        match self {
            CallableDef::FunctionId(f) => f.lookup(db).module(db),
            CallableDef::StructId(s) => s.lookup(db).container.module(db),
            CallableDef::EnumVariantId(e) => e.parent.lookup(db).container.module(db),
        }
        .krate
    }
}

impl From<CallableDef> for GenericDefId {
    fn from(def: CallableDef) -> GenericDefId {
        match def {
            CallableDef::FunctionId(f) => f.into(),
            CallableDef::StructId(s) => s.into(),
            CallableDef::EnumVariantId(e) => e.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyDefId {
    BuiltinType(BuiltinType),
    AdtId(AdtId),
    TypeAliasId(TypeAliasId),
}
impl_froms!(TyDefId: BuiltinType, AdtId(StructId, EnumId, UnionId), TypeAliasId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueTyDefId {
    FunctionId(FunctionId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
    ConstId(ConstId),
    StaticId(StaticId),
}
impl_froms!(ValueTyDefId: FunctionId, StructId, EnumVariantId, ConstId, StaticId);

/// Build the declared type of an item. This depends on the namespace; e.g. for
/// `struct Foo(usize)`, we have two types: The type of the struct itself, and
/// the constructor function `(usize) -> Foo` which lives in the values
/// namespace.
pub(crate) fn ty_query(db: &dyn HirDatabase, def: TyDefId) -> Binders<Ty> {
    match def {
        TyDefId::BuiltinType(it) => Binders::new(0, type_for_builtin(it)),
        TyDefId::AdtId(it) => type_for_adt(db, it),
        TyDefId::TypeAliasId(it) => type_for_type_alias(db, it),
    }
}

pub(crate) fn ty_recover(db: &dyn HirDatabase, _cycle: &[String], def: &TyDefId) -> Binders<Ty> {
    let num_binders = match *def {
        TyDefId::BuiltinType(_) => 0,
        TyDefId::AdtId(it) => generics(db.upcast(), it.into()).len(),
        TyDefId::TypeAliasId(it) => generics(db.upcast(), it.into()).len(),
    };
    Binders::new(num_binders, Ty::Unknown)
}

pub(crate) fn value_ty_query(db: &dyn HirDatabase, def: ValueTyDefId) -> Binders<Ty> {
    match def {
        ValueTyDefId::FunctionId(it) => type_for_fn(db, it),
        ValueTyDefId::StructId(it) => type_for_struct_constructor(db, it),
        ValueTyDefId::EnumVariantId(it) => type_for_enum_variant_constructor(db, it),
        ValueTyDefId::ConstId(it) => type_for_const(db, it),
        ValueTyDefId::StaticId(it) => type_for_static(db, it),
    }
}

pub(crate) fn impl_self_ty_query(db: &dyn HirDatabase, impl_id: ImplId) -> Binders<Ty> {
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db.upcast());
    let generics = generics(db.upcast(), impl_id.into());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    Binders::new(generics.len(), Ty::from_hir(&ctx, &impl_data.target_type))
}

pub(crate) fn impl_self_ty_recover(
    db: &dyn HirDatabase,
    _cycle: &[String],
    impl_id: &ImplId,
) -> Binders<Ty> {
    let generics = generics(db.upcast(), (*impl_id).into());
    Binders::new(generics.len(), Ty::Unknown)
}

pub(crate) fn impl_trait_query(db: &dyn HirDatabase, impl_id: ImplId) -> Option<Binders<TraitRef>> {
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let self_ty = db.impl_self_ty(impl_id);
    let target_trait = impl_data.target_trait.as_ref()?;
    Some(Binders::new(
        self_ty.num_binders,
        TraitRef::from_hir(&ctx, target_trait, Some(self_ty.value))?,
    ))
}
