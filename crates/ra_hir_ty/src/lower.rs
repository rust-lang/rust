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
    builtin_type::BuiltinType,
    generics::WherePredicate,
    path::{GenericArg, Path, PathSegment, PathSegments},
    resolver::{HasResolver, Resolver, TypeNs},
    type_ref::{TypeBound, TypeRef},
    AdtId, ConstId, EnumId, EnumVariantId, FunctionId, GenericDefId, HasModule, ImplId,
    LocalStructFieldId, Lookup, StaticId, StructId, TraitId, TypeAliasId, UnionId, VariantId,
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
    FnSig, GenericPredicate, ProjectionPredicate, ProjectionTy, Substs, TraitEnvironment, TraitRef,
    Ty, TypeCtor, TypeWalk,
};

impl Ty {
    pub fn from_hir(db: &impl HirDatabase, resolver: &Resolver, type_ref: &TypeRef) -> Self {
        match type_ref {
            TypeRef::Never => Ty::simple(TypeCtor::Never),
            TypeRef::Tuple(inner) => {
                let inner_tys: Arc<[Ty]> =
                    inner.iter().map(|tr| Ty::from_hir(db, resolver, tr)).collect();
                Ty::apply(
                    TypeCtor::Tuple { cardinality: inner_tys.len() as u16 },
                    Substs(inner_tys),
                )
            }
            TypeRef::Path(path) => Ty::from_hir_path(db, resolver, path),
            TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = Ty::from_hir(db, resolver, inner);
                Ty::apply_one(TypeCtor::RawPtr(*mutability), inner_ty)
            }
            TypeRef::Array(inner) => {
                let inner_ty = Ty::from_hir(db, resolver, inner);
                Ty::apply_one(TypeCtor::Array, inner_ty)
            }
            TypeRef::Slice(inner) => {
                let inner_ty = Ty::from_hir(db, resolver, inner);
                Ty::apply_one(TypeCtor::Slice, inner_ty)
            }
            TypeRef::Reference(inner, mutability) => {
                let inner_ty = Ty::from_hir(db, resolver, inner);
                Ty::apply_one(TypeCtor::Ref(*mutability), inner_ty)
            }
            TypeRef::Placeholder => Ty::Unknown,
            TypeRef::Fn(params) => {
                let sig = Substs(params.iter().map(|tr| Ty::from_hir(db, resolver, tr)).collect());
                Ty::apply(TypeCtor::FnPtr { num_args: sig.len() as u16 - 1 }, sig)
            }
            TypeRef::DynTrait(bounds) => {
                let self_ty = Ty::Bound(0);
                let predicates = bounds
                    .iter()
                    .flat_map(|b| {
                        GenericPredicate::from_type_bound(db, resolver, b, self_ty.clone())
                    })
                    .collect();
                Ty::Dyn(predicates)
            }
            TypeRef::ImplTrait(bounds) => {
                let self_ty = Ty::Bound(0);
                let predicates = bounds
                    .iter()
                    .flat_map(|b| {
                        GenericPredicate::from_type_bound(db, resolver, b, self_ty.clone())
                    })
                    .collect();
                Ty::Opaque(predicates)
            }
            TypeRef::Error => Ty::Unknown,
        }
    }

    /// This is only for `generic_predicates_for_param`, where we can't just
    /// lower the self types of the predicates since that could lead to cycles.
    /// So we just check here if the `type_ref` resolves to a generic param, and which.
    fn from_hir_only_param(
        db: &impl HirDatabase,
        resolver: &Resolver,
        type_ref: &TypeRef,
    ) -> Option<u32> {
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
        let resolution = match resolver.resolve_path_in_type_ns(db, path.mod_path()) {
            Some((it, None)) => it,
            _ => return None,
        };
        if let TypeNs::GenericParam(param_id) = resolution {
            let generics = generics(db, resolver.generic_def().expect("generics in scope"));
            let idx = generics.param_idx(param_id);
            Some(idx)
        } else {
            None
        }
    }

    pub(crate) fn from_type_relative_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        ty: Ty,
        remaining_segments: PathSegments<'_>,
    ) -> Ty {
        if remaining_segments.len() == 1 {
            // resolve unselected assoc types
            let segment = remaining_segments.first().unwrap();
            Ty::select_associated_type(db, resolver, ty, segment)
        } else if remaining_segments.len() > 1 {
            // FIXME report error (ambiguous associated type)
            Ty::Unknown
        } else {
            ty
        }
    }

    pub(crate) fn from_partly_resolved_hir_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        resolution: TypeNs,
        resolved_segment: PathSegment<'_>,
        remaining_segments: PathSegments<'_>,
    ) -> Ty {
        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                let trait_ref =
                    TraitRef::from_resolved_path(db, resolver, trait_, resolved_segment, None);
                return if remaining_segments.len() == 1 {
                    let segment = remaining_segments.first().unwrap();
                    let associated_ty = associated_type_by_name_including_super_traits(
                        db,
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
            }
            TypeNs::GenericParam(param_id) => {
                let generics = generics(db, resolver.generic_def().expect("generics in scope"));
                let idx = generics.param_idx(param_id);
                // FIXME: maybe return name in resolution?
                let name = generics.param_name(param_id);
                Ty::Param { idx, name }
            }
            TypeNs::SelfType(impl_id) => db.impl_self_ty(impl_id).clone(),
            TypeNs::AdtSelfType(adt) => db.ty(adt.into()),

            TypeNs::AdtId(it) => Ty::from_hir_path_inner(db, resolver, resolved_segment, it.into()),
            TypeNs::BuiltinType(it) => {
                Ty::from_hir_path_inner(db, resolver, resolved_segment, it.into())
            }
            TypeNs::TypeAliasId(it) => {
                Ty::from_hir_path_inner(db, resolver, resolved_segment, it.into())
            }
            // FIXME: report error
            TypeNs::EnumVariantId(_) => return Ty::Unknown,
        };

        Ty::from_type_relative_path(db, resolver, ty, remaining_segments)
    }

    pub(crate) fn from_hir_path(db: &impl HirDatabase, resolver: &Resolver, path: &Path) -> Ty {
        // Resolve the path (in type namespace)
        if let Some(type_ref) = path.type_anchor() {
            let ty = Ty::from_hir(db, resolver, &type_ref);
            return Ty::from_type_relative_path(db, resolver, ty, path.segments());
        }
        let (resolution, remaining_index) =
            match resolver.resolve_path_in_type_ns(db, path.mod_path()) {
                Some(it) => it,
                None => return Ty::Unknown,
            };
        let (resolved_segment, remaining_segments) = match remaining_index {
            None => (
                path.segments().last().expect("resolved path has at least one element"),
                PathSegments::EMPTY,
            ),
            Some(i) => (path.segments().get(i - 1).unwrap(), path.segments().skip(i)),
        };
        Ty::from_partly_resolved_hir_path(
            db,
            resolver,
            resolution,
            resolved_segment,
            remaining_segments,
        )
    }

    fn select_associated_type(
        db: &impl HirDatabase,
        resolver: &Resolver,
        self_ty: Ty,
        segment: PathSegment<'_>,
    ) -> Ty {
        let param_idx = match self_ty {
            Ty::Param { idx, .. } => idx,
            _ => return Ty::Unknown, // Error: Ambiguous associated type
        };
        let def = match resolver.generic_def() {
            Some(def) => def,
            None => return Ty::Unknown, // this can't actually happen
        };
        let predicates = db.generic_predicates_for_param(def.into(), param_idx);
        let traits_from_env = predicates.iter().filter_map(|pred| match pred {
            GenericPredicate::Implemented(tr) if tr.self_ty() == &self_ty => Some(tr.trait_),
            _ => None,
        });
        let traits = traits_from_env.flat_map(|t| all_super_traits(db, t));
        for t in traits {
            if let Some(associated_ty) = db.trait_data(t).associated_type_by_name(&segment.name) {
                let substs =
                    Substs::build_for_def(db, t).push(self_ty.clone()).fill_with_unknown().build();
                // FIXME handle type parameters on the segment
                return Ty::Projection(ProjectionTy { associated_ty, parameters: substs });
            }
        }
        Ty::Unknown
    }

    fn from_hir_path_inner(
        db: &impl HirDatabase,
        resolver: &Resolver,
        segment: PathSegment<'_>,
        typable: TyDefId,
    ) -> Ty {
        let generic_def = match typable {
            TyDefId::BuiltinType(_) => None,
            TyDefId::AdtId(it) => Some(it.into()),
            TyDefId::TypeAliasId(it) => Some(it.into()),
        };
        let substs = substs_from_path_segment(db, resolver, segment, generic_def, false);
        db.ty(typable).subst(&substs)
    }

    /// Collect generic arguments from a path into a `Substs`. See also
    /// `create_substs_for_ast_path` and `def_to_ty` in rustc.
    pub(super) fn substs_from_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
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
        substs_from_path_segment(db, resolver, segment, generic_def, false)
    }
}

pub(super) fn substs_from_path_segment(
    db: &impl HirDatabase,
    resolver: &Resolver,
    segment: PathSegment<'_>,
    def_generic: Option<GenericDefId>,
    add_self_param: bool,
) -> Substs {
    let mut substs = Vec::new();
    let def_generics = def_generic.map(|def| generics(db, def.into()));

    let (total_len, parent_len, child_len) = def_generics.map_or((0, 0, 0), |g| g.len_split());
    substs.extend(iter::repeat(Ty::Unknown).take(parent_len));
    if add_self_param {
        // FIXME this add_self_param argument is kind of a hack: Traits have the
        // Self type as an implicit first type parameter, but it can't be
        // actually provided in the type arguments
        // (well, actually sometimes it can, in the form of type-relative paths: `<Foo as Default>::default()`)
        substs.push(Ty::Unknown);
    }
    if let Some(generic_args) = &segment.args_and_bindings {
        // if args are provided, it should be all of them, but we can't rely on that
        let self_param_correction = if add_self_param { 1 } else { 0 };
        let child_len = child_len + self_param_correction;
        for arg in generic_args.args.iter().take(child_len) {
            match arg {
                GenericArg::Type(type_ref) => {
                    let ty = Ty::from_hir(db, resolver, type_ref);
                    substs.push(ty);
                }
            }
        }
    }
    // add placeholders for args that were not provided
    let supplied_params = substs.len();
    for _ in supplied_params..total_len {
        substs.push(Ty::Unknown);
    }
    assert_eq!(substs.len(), total_len);

    // handle defaults
    if let Some(def_generic) = def_generic {
        let default_substs = db.generic_defaults(def_generic.into());
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
        db: &impl HirDatabase,
        resolver: &Resolver,
        path: &Path,
        explicit_self_ty: Option<Ty>,
    ) -> Option<Self> {
        let resolved = match resolver.resolve_path_in_type_ns_fully(db, path.mod_path())? {
            TypeNs::TraitId(tr) => tr,
            _ => return None,
        };
        let segment = path.segments().last().expect("path should have at least one segment");
        Some(TraitRef::from_resolved_path(db, resolver, resolved.into(), segment, explicit_self_ty))
    }

    pub(crate) fn from_resolved_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        resolved: TraitId,
        segment: PathSegment<'_>,
        explicit_self_ty: Option<Ty>,
    ) -> Self {
        let mut substs = TraitRef::substs_from_path(db, resolver, segment, resolved);
        if let Some(self_ty) = explicit_self_ty {
            make_mut_slice(&mut substs.0)[0] = self_ty;
        }
        TraitRef { trait_: resolved, substs }
    }

    fn from_hir(
        db: &impl HirDatabase,
        resolver: &Resolver,
        type_ref: &TypeRef,
        explicit_self_ty: Option<Ty>,
    ) -> Option<Self> {
        let path = match type_ref {
            TypeRef::Path(path) => path,
            _ => return None,
        };
        TraitRef::from_path(db, resolver, path, explicit_self_ty)
    }

    fn substs_from_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        segment: PathSegment<'_>,
        resolved: TraitId,
    ) -> Substs {
        let has_self_param =
            segment.args_and_bindings.as_ref().map(|a| a.has_self_type).unwrap_or(false);
        substs_from_path_segment(db, resolver, segment, Some(resolved.into()), !has_self_param)
    }

    pub(crate) fn from_type_bound(
        db: &impl HirDatabase,
        resolver: &Resolver,
        bound: &TypeBound,
        self_ty: Ty,
    ) -> Option<TraitRef> {
        match bound {
            TypeBound::Path(path) => TraitRef::from_path(db, resolver, path, Some(self_ty)),
            TypeBound::Error => None,
        }
    }
}

impl GenericPredicate {
    pub(crate) fn from_where_predicate<'a>(
        db: &'a impl HirDatabase,
        resolver: &'a Resolver,
        where_predicate: &'a WherePredicate,
    ) -> impl Iterator<Item = GenericPredicate> + 'a {
        let self_ty = Ty::from_hir(db, resolver, &where_predicate.type_ref);
        GenericPredicate::from_type_bound(db, resolver, &where_predicate.bound, self_ty)
    }

    pub(crate) fn from_type_bound<'a>(
        db: &'a impl HirDatabase,
        resolver: &'a Resolver,
        bound: &'a TypeBound,
        self_ty: Ty,
    ) -> impl Iterator<Item = GenericPredicate> + 'a {
        let trait_ref = TraitRef::from_type_bound(db, &resolver, bound, self_ty);
        iter::once(trait_ref.clone().map_or(GenericPredicate::Error, GenericPredicate::Implemented))
            .chain(
                trait_ref.into_iter().flat_map(move |tr| {
                    assoc_type_bindings_from_type_bound(db, resolver, bound, tr)
                }),
            )
    }
}

fn assoc_type_bindings_from_type_bound<'a>(
    db: &'a impl HirDatabase,
    resolver: &'a Resolver,
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
            let associated_ty =
                associated_type_by_name_including_super_traits(db, trait_ref.trait_, &name);
            let associated_ty = match associated_ty {
                None => return GenericPredicate::Error,
                Some(t) => t,
            };
            let projection_ty =
                ProjectionTy { associated_ty, parameters: trait_ref.substs.clone() };
            let ty = Ty::from_hir(db, resolver, type_ref);
            let projection_predicate = ProjectionPredicate { projection_ty, ty };
            GenericPredicate::Projection(projection_predicate)
        })
}

/// Build the signature of a callable item (function, struct or enum variant).
pub fn callable_item_sig(db: &impl HirDatabase, def: CallableDef) -> FnSig {
    match def {
        CallableDef::FunctionId(f) => fn_sig_for_fn(db, f),
        CallableDef::StructId(s) => fn_sig_for_struct_constructor(db, s),
        CallableDef::EnumVariantId(e) => fn_sig_for_enum_variant_constructor(db, e),
    }
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_query(
    db: &impl HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalStructFieldId, Ty>> {
    let var_data = variant_data(db, variant_id);
    let resolver = match variant_id {
        VariantId::StructId(it) => it.resolver(db),
        VariantId::UnionId(it) => it.resolver(db),
        VariantId::EnumVariantId(it) => it.parent.resolver(db),
    };
    let mut res = ArenaMap::default();
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, Ty::from_hir(db, &resolver, &field_data.type_ref))
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
    db: &impl HirDatabase,
    def: GenericDefId,
    param_idx: u32,
) -> Arc<[GenericPredicate]> {
    let resolver = def.resolver(db);
    resolver
        .where_predicates_in_scope()
        // we have to filter out all other predicates *first*, before attempting to lower them
        .filter(|pred| Ty::from_hir_only_param(db, &resolver, &pred.type_ref) == Some(param_idx))
        .flat_map(|pred| GenericPredicate::from_where_predicate(db, &resolver, pred))
        .collect()
}

pub(crate) fn generic_predicates_for_param_recover(
    _db: &impl HirDatabase,
    _cycle: &[String],
    _def: &GenericDefId,
    _param_idx: &u32,
) -> Arc<[GenericPredicate]> {
    Arc::new([])
}

impl TraitEnvironment {
    pub fn lower(db: &impl HirDatabase, resolver: &Resolver) -> Arc<TraitEnvironment> {
        let predicates = resolver
            .where_predicates_in_scope()
            .flat_map(|pred| GenericPredicate::from_where_predicate(db, &resolver, pred))
            .collect::<Vec<_>>();

        Arc::new(TraitEnvironment { predicates })
    }
}

/// Resolve the where clause(s) of an item with generics.
pub(crate) fn generic_predicates_query(
    db: &impl HirDatabase,
    def: GenericDefId,
) -> Arc<[GenericPredicate]> {
    let resolver = def.resolver(db);
    resolver
        .where_predicates_in_scope()
        .flat_map(|pred| GenericPredicate::from_where_predicate(db, &resolver, pred))
        .collect()
}

/// Resolve the default type params from generics
pub(crate) fn generic_defaults_query(db: &impl HirDatabase, def: GenericDefId) -> Substs {
    let resolver = def.resolver(db);
    let generic_params = generics(db, def.into());

    let defaults = generic_params
        .iter()
        .map(|(_idx, p)| p.default.as_ref().map_or(Ty::Unknown, |t| Ty::from_hir(db, &resolver, t)))
        .collect();

    Substs(defaults)
}

fn fn_sig_for_fn(db: &impl HirDatabase, def: FunctionId) -> FnSig {
    let data = db.function_data(def);
    let resolver = def.resolver(db);
    let params = data.params.iter().map(|tr| Ty::from_hir(db, &resolver, tr)).collect::<Vec<_>>();
    let ret = Ty::from_hir(db, &resolver, &data.ret_type);
    FnSig::from_params_and_return(params, ret)
}

/// Build the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn(db: &impl HirDatabase, def: FunctionId) -> Ty {
    let generics = generics(db, def.into());
    let substs = Substs::identity(&generics);
    Ty::apply(TypeCtor::FnDef(def.into()), substs)
}

/// Build the declared type of a const.
fn type_for_const(db: &impl HirDatabase, def: ConstId) -> Ty {
    let data = db.const_data(def);
    let resolver = def.resolver(db);

    Ty::from_hir(db, &resolver, &data.type_ref)
}

/// Build the declared type of a static.
fn type_for_static(db: &impl HirDatabase, def: StaticId) -> Ty {
    let data = db.static_data(def);
    let resolver = def.resolver(db);

    Ty::from_hir(db, &resolver, &data.type_ref)
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

fn fn_sig_for_struct_constructor(db: &impl HirDatabase, def: StructId) -> FnSig {
    let struct_data = db.struct_data(def.into());
    let fields = struct_data.variant_data.fields();
    let resolver = def.resolver(db);
    let params = fields
        .iter()
        .map(|(_, field)| Ty::from_hir(db, &resolver, &field.type_ref))
        .collect::<Vec<_>>();
    let ret = type_for_adt(db, def.into());
    FnSig::from_params_and_return(params, ret)
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor(db: &impl HirDatabase, def: StructId) -> Ty {
    let struct_data = db.struct_data(def.into());
    if struct_data.variant_data.is_unit() {
        return type_for_adt(db, def.into()); // Unit struct
    }
    let generics = generics(db, def.into());
    let substs = Substs::identity(&generics);
    Ty::apply(TypeCtor::FnDef(def.into()), substs)
}

fn fn_sig_for_enum_variant_constructor(db: &impl HirDatabase, def: EnumVariantId) -> FnSig {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id];
    let fields = var_data.variant_data.fields();
    let resolver = def.parent.resolver(db);
    let params = fields
        .iter()
        .map(|(_, field)| Ty::from_hir(db, &resolver, &field.type_ref))
        .collect::<Vec<_>>();
    let generics = generics(db, def.parent.into());
    let substs = Substs::identity(&generics);
    let ret = type_for_adt(db, def.parent.into()).subst(&substs);
    FnSig::from_params_and_return(params, ret)
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor(db: &impl HirDatabase, def: EnumVariantId) -> Ty {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id].variant_data;
    if var_data.is_unit() {
        return type_for_adt(db, def.parent.into()); // Unit variant
    }
    let generics = generics(db, def.parent.into());
    let substs = Substs::identity(&generics);
    Ty::apply(TypeCtor::FnDef(EnumVariantId::from(def).into()), substs)
}

fn type_for_adt(db: &impl HirDatabase, adt: AdtId) -> Ty {
    let generics = generics(db, adt.into());
    Ty::apply(TypeCtor::Adt(adt), Substs::identity(&generics))
}

fn type_for_type_alias(db: &impl HirDatabase, t: TypeAliasId) -> Ty {
    let generics = generics(db, t.into());
    let resolver = t.resolver(db);
    let type_ref = &db.type_alias_data(t).type_ref;
    let substs = Substs::identity(&generics);
    let inner = Ty::from_hir(db, &resolver, type_ref.as_ref().unwrap_or(&TypeRef::Error));
    inner.subst(&substs)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableDef {
    FunctionId(FunctionId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
}
impl_froms!(CallableDef: FunctionId, StructId, EnumVariantId);

impl CallableDef {
    pub fn krate(self, db: &impl HirDatabase) -> CrateId {
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
pub(crate) fn ty_query(db: &impl HirDatabase, def: TyDefId) -> Ty {
    match def {
        TyDefId::BuiltinType(it) => type_for_builtin(it),
        TyDefId::AdtId(it) => type_for_adt(db, it),
        TyDefId::TypeAliasId(it) => type_for_type_alias(db, it),
    }
}

pub(crate) fn ty_recover(_db: &impl HirDatabase, _cycle: &[String], _def: &TyDefId) -> Ty {
    Ty::Unknown
}

pub(crate) fn value_ty_query(db: &impl HirDatabase, def: ValueTyDefId) -> Ty {
    match def {
        ValueTyDefId::FunctionId(it) => type_for_fn(db, it),
        ValueTyDefId::StructId(it) => type_for_struct_constructor(db, it),
        ValueTyDefId::EnumVariantId(it) => type_for_enum_variant_constructor(db, it),
        ValueTyDefId::ConstId(it) => type_for_const(db, it),
        ValueTyDefId::StaticId(it) => type_for_static(db, it),
    }
}

pub(crate) fn impl_self_ty_query(db: &impl HirDatabase, impl_id: ImplId) -> Ty {
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db);
    Ty::from_hir(db, &resolver, &impl_data.target_type)
}

pub(crate) fn impl_self_ty_recover(
    _db: &impl HirDatabase,
    _cycle: &[String],
    _impl_id: &ImplId,
) -> Ty {
    Ty::Unknown
}

pub(crate) fn impl_trait_query(db: &impl HirDatabase, impl_id: ImplId) -> Option<TraitRef> {
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db);
    let self_ty = db.impl_self_ty(impl_id);
    let target_trait = impl_data.target_trait.as_ref()?;
    TraitRef::from_hir(db, &resolver, target_trait, Some(self_ty.clone()))
}
