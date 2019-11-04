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
    path::{GenericArg, PathSegment},
    type_ref::{TypeBound, TypeRef},
};

use super::{
    FnSig, GenericPredicate, ProjectionPredicate, ProjectionTy, Substs, TraitRef, Ty, TypeCtor,
    TypeWalk,
};
use crate::{
    adt::VariantDef,
    db::HirDatabase,
    generics::HasGenericParams,
    generics::{GenericDef, WherePredicate},
    resolve::{Resolver, TypeNs},
    ty::{
        primitive::{FloatTy, IntTy},
        Adt,
    },
    util::make_mut_slice,
    Const, Enum, EnumVariant, Function, ModuleDef, Namespace, Path, Static, Struct, StructField,
    Trait, TypeAlias, Union,
};

impl Ty {
    pub(crate) fn from_hir(db: &impl HirDatabase, resolver: &Resolver, type_ref: &TypeRef) -> Self {
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
        if let crate::PathKind::Type(_) = &path.kind {
            return None;
        }
        if path.segments.len() > 1 {
            return None;
        }
        let resolution = match resolver.resolve_path_in_type_ns(db, path) {
            Some((it, None)) => it,
            _ => return None,
        };
        if let TypeNs::GenericParam(idx) = resolution {
            Some(idx)
        } else {
            None
        }
    }

    pub(crate) fn from_type_relative_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        ty: Ty,
        remaining_segments: &[PathSegment],
    ) -> Ty {
        if remaining_segments.len() == 1 {
            // resolve unselected assoc types
            let segment = &remaining_segments[0];
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
        resolved_segment: &PathSegment,
        remaining_segments: &[PathSegment],
    ) -> Ty {
        let ty = match resolution {
            TypeNs::Trait(trait_) => {
                let trait_ref =
                    TraitRef::from_resolved_path(db, resolver, trait_, resolved_segment, None);
                return if remaining_segments.len() == 1 {
                    let segment = &remaining_segments[0];
                    match trait_ref
                        .trait_
                        .associated_type_by_name_including_super_traits(db, &segment.name)
                    {
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
            TypeNs::GenericParam(idx) => {
                // FIXME: maybe return name in resolution?
                let name = resolved_segment.name.clone();
                Ty::Param { idx, name }
            }
            TypeNs::SelfType(impl_block) => impl_block.target_ty(db),
            TypeNs::AdtSelfType(adt) => adt.ty(db),

            TypeNs::Adt(it) => Ty::from_hir_path_inner(db, resolver, resolved_segment, it.into()),
            TypeNs::BuiltinType(it) => {
                Ty::from_hir_path_inner(db, resolver, resolved_segment, it.into())
            }
            TypeNs::TypeAlias(it) => {
                Ty::from_hir_path_inner(db, resolver, resolved_segment, it.into())
            }
            // FIXME: report error
            TypeNs::EnumVariant(_) => return Ty::Unknown,
        };

        Ty::from_type_relative_path(db, resolver, ty, remaining_segments)
    }

    pub(crate) fn from_hir_path(db: &impl HirDatabase, resolver: &Resolver, path: &Path) -> Ty {
        // Resolve the path (in type namespace)
        if let crate::PathKind::Type(type_ref) = &path.kind {
            let ty = Ty::from_hir(db, resolver, &type_ref);
            let remaining_segments = &path.segments[..];
            return Ty::from_type_relative_path(db, resolver, ty, remaining_segments);
        }
        let (resolution, remaining_index) = match resolver.resolve_path_in_type_ns(db, path) {
            Some(it) => it,
            None => return Ty::Unknown,
        };
        let (resolved_segment, remaining_segments) = match remaining_index {
            None => (
                path.segments.last().expect("resolved path has at least one element"),
                &[] as &[PathSegment],
            ),
            Some(i) => (&path.segments[i - 1], &path.segments[i..]),
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
        segment: &PathSegment,
    ) -> Ty {
        let param_idx = match self_ty {
            Ty::Param { idx, .. } => idx,
            _ => return Ty::Unknown, // Error: Ambiguous associated type
        };
        let def = match resolver.generic_def() {
            Some(def) => def,
            None => return Ty::Unknown, // this can't actually happen
        };
        let predicates = db.generic_predicates_for_param(def, param_idx);
        let traits_from_env = predicates.iter().filter_map(|pred| match pred {
            GenericPredicate::Implemented(tr) if tr.self_ty() == &self_ty => Some(tr.trait_),
            _ => None,
        });
        let traits = traits_from_env.flat_map(|t| t.all_super_traits(db));
        for t in traits {
            if let Some(associated_ty) = t.associated_type_by_name(db, &segment.name) {
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
        segment: &PathSegment,
        typable: TypableDef,
    ) -> Ty {
        let ty = db.type_for_def(typable, Namespace::Types);
        let substs = Ty::substs_from_path_segment(db, resolver, segment, typable);
        ty.subst(&substs)
    }

    pub(super) fn substs_from_path_segment(
        db: &impl HirDatabase,
        resolver: &Resolver,
        segment: &PathSegment,
        resolved: TypableDef,
    ) -> Substs {
        let def_generic: Option<GenericDef> = match resolved {
            TypableDef::Function(func) => Some(func.into()),
            TypableDef::Adt(adt) => Some(adt.into()),
            TypableDef::EnumVariant(var) => Some(var.parent_enum(db).into()),
            TypableDef::TypeAlias(t) => Some(t.into()),
            TypableDef::Const(_) | TypableDef::Static(_) | TypableDef::BuiltinType(_) => None,
        };
        substs_from_path_segment(db, resolver, segment, def_generic, false)
    }

    /// Collect generic arguments from a path into a `Substs`. See also
    /// `create_substs_for_ast_path` and `def_to_ty` in rustc.
    pub(super) fn substs_from_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        path: &Path,
        resolved: TypableDef,
    ) -> Substs {
        let last = path.segments.last().expect("path should have at least one segment");
        let segment = match resolved {
            TypableDef::Function(_)
            | TypableDef::Adt(_)
            | TypableDef::Const(_)
            | TypableDef::Static(_)
            | TypableDef::TypeAlias(_)
            | TypableDef::BuiltinType(_) => last,
            TypableDef::EnumVariant(_) => {
                // the generic args for an enum variant may be either specified
                // on the segment referring to the enum, or on the segment
                // referring to the variant. So `Option::<T>::None` and
                // `Option::None::<T>` are both allowed (though the former is
                // preferred). See also `def_ids_for_path_segments` in rustc.
                let len = path.segments.len();
                let segment = if len >= 2 && path.segments[len - 2].args_and_bindings.is_some() {
                    // Option::<T>::None
                    &path.segments[len - 2]
                } else {
                    // Option::None::<T>
                    last
                };
                segment
            }
        };
        Ty::substs_from_path_segment(db, resolver, segment, resolved)
    }
}

pub(super) fn substs_from_path_segment(
    db: &impl HirDatabase,
    resolver: &Resolver,
    segment: &PathSegment,
    def_generic: Option<GenericDef>,
    add_self_param: bool,
) -> Substs {
    let mut substs = Vec::new();
    let def_generics = def_generic.map(|def| def.generic_params(db));

    let (parent_param_count, param_count) =
        def_generics.map_or((0, 0), |g| (g.count_parent_params(), g.params.len()));
    substs.extend(iter::repeat(Ty::Unknown).take(parent_param_count));
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
        let param_count = param_count - self_param_correction;
        for arg in generic_args.args.iter().take(param_count) {
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
    for _ in supplied_params..parent_param_count + param_count {
        substs.push(Ty::Unknown);
    }
    assert_eq!(substs.len(), parent_param_count + param_count);

    // handle defaults
    if let Some(def_generic) = def_generic {
        let default_substs = db.generic_defaults(def_generic);
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
    pub(crate) fn from_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        path: &Path,
        explicit_self_ty: Option<Ty>,
    ) -> Option<Self> {
        let resolved = match resolver.resolve_path_in_type_ns_fully(db, &path)? {
            TypeNs::Trait(tr) => tr,
            _ => return None,
        };
        let segment = path.segments.last().expect("path should have at least one segment");
        Some(TraitRef::from_resolved_path(db, resolver, resolved, segment, explicit_self_ty))
    }

    pub(super) fn from_resolved_path(
        db: &impl HirDatabase,
        resolver: &Resolver,
        resolved: Trait,
        segment: &PathSegment,
        explicit_self_ty: Option<Ty>,
    ) -> Self {
        let mut substs = TraitRef::substs_from_path(db, resolver, segment, resolved);
        if let Some(self_ty) = explicit_self_ty {
            make_mut_slice(&mut substs.0)[0] = self_ty;
        }
        TraitRef { trait_: resolved, substs }
    }

    pub(crate) fn from_hir(
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
        segment: &PathSegment,
        resolved: Trait,
    ) -> Substs {
        let has_self_param =
            segment.args_and_bindings.as_ref().map(|a| a.has_self_type).unwrap_or(false);
        substs_from_path_segment(db, resolver, segment, Some(resolved.into()), !has_self_param)
    }

    pub(crate) fn for_trait(db: &impl HirDatabase, trait_: Trait) -> TraitRef {
        let substs = Substs::identity(&trait_.generic_params(db));
        TraitRef { trait_, substs }
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
        TypeBound::Path(path) => path.segments.last(),
        TypeBound::Error => None,
    };
    last_segment
        .into_iter()
        .flat_map(|segment| segment.args_and_bindings.iter())
        .flat_map(|args_and_bindings| args_and_bindings.bindings.iter())
        .map(move |(name, type_ref)| {
            let associated_ty =
                match trait_ref.trait_.associated_type_by_name_including_super_traits(db, &name) {
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

/// Build the declared type of an item. This depends on the namespace; e.g. for
/// `struct Foo(usize)`, we have two types: The type of the struct itself, and
/// the constructor function `(usize) -> Foo` which lives in the values
/// namespace.
pub(crate) fn type_for_def(db: &impl HirDatabase, def: TypableDef, ns: Namespace) -> Ty {
    match (def, ns) {
        (TypableDef::Function(f), Namespace::Values) => type_for_fn(db, f),
        (TypableDef::Adt(Adt::Struct(s)), Namespace::Values) => type_for_struct_constructor(db, s),
        (TypableDef::Adt(adt), Namespace::Types) => type_for_adt(db, adt),
        (TypableDef::EnumVariant(v), Namespace::Values) => type_for_enum_variant_constructor(db, v),
        (TypableDef::TypeAlias(t), Namespace::Types) => type_for_type_alias(db, t),
        (TypableDef::Const(c), Namespace::Values) => type_for_const(db, c),
        (TypableDef::Static(c), Namespace::Values) => type_for_static(db, c),
        (TypableDef::BuiltinType(t), Namespace::Types) => type_for_builtin(t),

        // 'error' cases:
        (TypableDef::Function(_), Namespace::Types) => Ty::Unknown,
        (TypableDef::Adt(Adt::Union(_)), Namespace::Values) => Ty::Unknown,
        (TypableDef::Adt(Adt::Enum(_)), Namespace::Values) => Ty::Unknown,
        (TypableDef::EnumVariant(_), Namespace::Types) => Ty::Unknown,
        (TypableDef::TypeAlias(_), Namespace::Values) => Ty::Unknown,
        (TypableDef::Const(_), Namespace::Types) => Ty::Unknown,
        (TypableDef::Static(_), Namespace::Types) => Ty::Unknown,
        (TypableDef::BuiltinType(_), Namespace::Values) => Ty::Unknown,
    }
}

/// Build the signature of a callable item (function, struct or enum variant).
pub(crate) fn callable_item_sig(db: &impl HirDatabase, def: CallableDef) -> FnSig {
    match def {
        CallableDef::Function(f) => fn_sig_for_fn(db, f),
        CallableDef::Struct(s) => fn_sig_for_struct_constructor(db, s),
        CallableDef::EnumVariant(e) => fn_sig_for_enum_variant_constructor(db, e),
    }
}

/// Build the type of a specific field of a struct or enum variant.
pub(crate) fn type_for_field(db: &impl HirDatabase, field: StructField) -> Ty {
    let parent_def = field.parent_def(db);
    let resolver = match parent_def {
        VariantDef::Struct(it) => it.resolver(db),
        VariantDef::EnumVariant(it) => it.parent_enum(db).resolver(db),
    };
    let var_data = parent_def.variant_data(db);
    let type_ref = &var_data.fields().unwrap()[field.id].type_ref;
    Ty::from_hir(db, &resolver, type_ref)
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
    def: GenericDef,
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

pub(crate) fn trait_env(
    db: &impl HirDatabase,
    resolver: &Resolver,
) -> Arc<super::TraitEnvironment> {
    let predicates = resolver
        .where_predicates_in_scope()
        .flat_map(|pred| GenericPredicate::from_where_predicate(db, &resolver, pred))
        .collect::<Vec<_>>();

    Arc::new(super::TraitEnvironment { predicates })
}

/// Resolve the where clause(s) of an item with generics.
pub(crate) fn generic_predicates_query(
    db: &impl HirDatabase,
    def: GenericDef,
) -> Arc<[GenericPredicate]> {
    let resolver = def.resolver(db);
    resolver
        .where_predicates_in_scope()
        .flat_map(|pred| GenericPredicate::from_where_predicate(db, &resolver, pred))
        .collect()
}

/// Resolve the default type params from generics
pub(crate) fn generic_defaults_query(db: &impl HirDatabase, def: GenericDef) -> Substs {
    let resolver = def.resolver(db);
    let generic_params = def.generic_params(db);

    let defaults = generic_params
        .params_including_parent()
        .into_iter()
        .map(|p| {
            p.default.as_ref().map_or(Ty::Unknown, |path| Ty::from_hir_path(db, &resolver, path))
        })
        .collect();

    Substs(defaults)
}

fn fn_sig_for_fn(db: &impl HirDatabase, def: Function) -> FnSig {
    let data = def.data(db);
    let resolver = def.resolver(db);
    let params = data.params().iter().map(|tr| Ty::from_hir(db, &resolver, tr)).collect::<Vec<_>>();
    let ret = Ty::from_hir(db, &resolver, data.ret_type());
    FnSig::from_params_and_return(params, ret)
}

/// Build the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn(db: &impl HirDatabase, def: Function) -> Ty {
    let generics = def.generic_params(db);
    let substs = Substs::identity(&generics);
    Ty::apply(TypeCtor::FnDef(def.into()), substs)
}

/// Build the declared type of a const.
fn type_for_const(db: &impl HirDatabase, def: Const) -> Ty {
    let data = def.data(db);
    let resolver = def.resolver(db);

    Ty::from_hir(db, &resolver, data.type_ref())
}

/// Build the declared type of a static.
fn type_for_static(db: &impl HirDatabase, def: Static) -> Ty {
    let data = def.data(db);
    let resolver = def.resolver(db);

    Ty::from_hir(db, &resolver, data.type_ref())
}

/// Build the declared type of a static.
fn type_for_builtin(def: BuiltinType) -> Ty {
    Ty::simple(match def {
        BuiltinType::Char => TypeCtor::Char,
        BuiltinType::Bool => TypeCtor::Bool,
        BuiltinType::Str => TypeCtor::Str,
        BuiltinType::Int { signedness, bitness } => {
            TypeCtor::Int(IntTy { signedness, bitness }.into())
        }
        BuiltinType::Float { bitness } => TypeCtor::Float(FloatTy { bitness }.into()),
    })
}

fn fn_sig_for_struct_constructor(db: &impl HirDatabase, def: Struct) -> FnSig {
    let struct_data = db.struct_data(def.id);
    let fields = match struct_data.variant_data.fields() {
        Some(fields) => fields,
        None => panic!("fn_sig_for_struct_constructor called on unit struct"),
    };
    let resolver = def.resolver(db);
    let params = fields
        .iter()
        .map(|(_, field)| Ty::from_hir(db, &resolver, &field.type_ref))
        .collect::<Vec<_>>();
    let ret = type_for_adt(db, def);
    FnSig::from_params_and_return(params, ret)
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor(db: &impl HirDatabase, def: Struct) -> Ty {
    let struct_data = db.struct_data(def.id);
    if struct_data.variant_data.fields().is_none() {
        return type_for_adt(db, def); // Unit struct
    }
    let generics = def.generic_params(db);
    let substs = Substs::identity(&generics);
    Ty::apply(TypeCtor::FnDef(def.into()), substs)
}

fn fn_sig_for_enum_variant_constructor(db: &impl HirDatabase, def: EnumVariant) -> FnSig {
    let var_data = def.variant_data(db);
    let fields = match var_data.fields() {
        Some(fields) => fields,
        None => panic!("fn_sig_for_enum_variant_constructor called for unit variant"),
    };
    let resolver = def.parent_enum(db).resolver(db);
    let params = fields
        .iter()
        .map(|(_, field)| Ty::from_hir(db, &resolver, &field.type_ref))
        .collect::<Vec<_>>();
    let generics = def.parent_enum(db).generic_params(db);
    let substs = Substs::identity(&generics);
    let ret = type_for_adt(db, def.parent_enum(db)).subst(&substs);
    FnSig::from_params_and_return(params, ret)
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor(db: &impl HirDatabase, def: EnumVariant) -> Ty {
    let var_data = def.variant_data(db);
    if var_data.fields().is_none() {
        return type_for_adt(db, def.parent_enum(db)); // Unit variant
    }
    let generics = def.parent_enum(db).generic_params(db);
    let substs = Substs::identity(&generics);
    Ty::apply(TypeCtor::FnDef(def.into()), substs)
}

fn type_for_adt(db: &impl HirDatabase, adt: impl Into<Adt> + HasGenericParams) -> Ty {
    let generics = adt.generic_params(db);
    Ty::apply(TypeCtor::Adt(adt.into()), Substs::identity(&generics))
}

fn type_for_type_alias(db: &impl HirDatabase, t: TypeAlias) -> Ty {
    let generics = t.generic_params(db);
    let resolver = t.resolver(db);
    let type_ref = t.type_ref(db);
    let substs = Substs::identity(&generics);
    let inner = Ty::from_hir(db, &resolver, &type_ref.unwrap_or(TypeRef::Error));
    inner.subst(&substs)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TypableDef {
    Function(Function),
    Adt(Adt),
    EnumVariant(EnumVariant),
    TypeAlias(TypeAlias),
    Const(Const),
    Static(Static),
    BuiltinType(BuiltinType),
}
impl_froms!(
    TypableDef: Function,
    Adt(Struct, Enum, Union),
    EnumVariant,
    TypeAlias,
    Const,
    Static,
    BuiltinType
);

impl From<ModuleDef> for Option<TypableDef> {
    fn from(def: ModuleDef) -> Option<TypableDef> {
        let res = match def {
            ModuleDef::Function(f) => f.into(),
            ModuleDef::Adt(adt) => adt.into(),
            ModuleDef::EnumVariant(v) => v.into(),
            ModuleDef::TypeAlias(t) => t.into(),
            ModuleDef::Const(v) => v.into(),
            ModuleDef::Static(v) => v.into(),
            ModuleDef::BuiltinType(t) => t.into(),
            ModuleDef::Module(_) | ModuleDef::Trait(_) => return None,
        };
        Some(res)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableDef {
    Function(Function),
    Struct(Struct),
    EnumVariant(EnumVariant),
}
impl_froms!(CallableDef: Function, Struct, EnumVariant);

impl CallableDef {
    pub fn krate(self, db: &impl HirDatabase) -> Option<crate::Crate> {
        match self {
            CallableDef::Function(f) => f.krate(db),
            CallableDef::Struct(s) => s.krate(db),
            CallableDef::EnumVariant(e) => e.parent_enum(db).krate(db),
        }
    }
}

impl From<CallableDef> for GenericDef {
    fn from(def: CallableDef) -> GenericDef {
        match def {
            CallableDef::Function(f) => f.into(),
            CallableDef::Struct(s) => s.into(),
            CallableDef::EnumVariant(e) => e.into(),
        }
    }
}
