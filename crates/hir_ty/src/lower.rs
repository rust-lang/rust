//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `Ty::from_hir`.
//!  - Building the type for an item: This happens through the `type_for_def` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
use std::{iter, sync::Arc};

use base_db::CrateId;
use chalk_ir::Mutability;
use hir_def::{
    adt::StructKind,
    builtin_type::BuiltinType,
    generics::{TypeParamProvenance, WherePredicate, WherePredicateTypeTarget},
    path::{GenericArg, Path, PathSegment, PathSegments},
    resolver::{HasResolver, Resolver, TypeNs},
    type_ref::{TypeBound, TypeRef},
    AdtId, AssocContainerId, AssocItemId, ConstId, ConstParamId, EnumId, EnumVariantId, FunctionId,
    GenericDefId, HasModule, ImplId, LocalFieldId, Lookup, StaticId, StructId, TraitId,
    TypeAliasId, TypeParamId, UnionId, VariantId,
};
use hir_expand::name::Name;
use la_arena::ArenaMap;
use smallvec::SmallVec;
use stdx::impl_from;

use crate::{
    db::HirDatabase,
    utils::{
        all_super_trait_refs, associated_type_by_name_including_super_traits, generics,
        make_mut_slice, variant_data,
    },
    AliasTy, Binders, BoundVar, CallableSig, DebruijnIndex, FnPointer, FnSig, GenericPredicate,
    OpaqueTy, OpaqueTyId, PolyFnSig, ProjectionPredicate, ProjectionTy, ReturnTypeImplTrait,
    ReturnTypeImplTraits, Substs, TraitEnvironment, TraitRef, Ty, TypeWalk,
};

#[derive(Debug)]
pub struct TyLoweringContext<'a> {
    pub db: &'a dyn HirDatabase,
    pub resolver: &'a Resolver,
    in_binders: DebruijnIndex,
    /// Note: Conceptually, it's thinkable that we could be in a location where
    /// some type params should be represented as placeholders, and others
    /// should be converted to variables. I think in practice, this isn't
    /// possible currently, so this should be fine for now.
    pub type_param_mode: TypeParamLoweringMode,
    pub impl_trait_mode: ImplTraitLoweringMode,
    impl_trait_counter: std::cell::Cell<u16>,
    /// When turning `impl Trait` into opaque types, we have to collect the
    /// bounds at the same time to get the IDs correct (without becoming too
    /// complicated). I don't like using interior mutability (as for the
    /// counter), but I've tried and failed to make the lifetimes work for
    /// passing around a `&mut TyLoweringContext`. The core problem is that
    /// we're grouping the mutable data (the counter and this field) together
    /// with the immutable context (the references to the DB and resolver).
    /// Splitting this up would be a possible fix.
    opaque_type_data: std::cell::RefCell<Vec<ReturnTypeImplTrait>>,
}

impl<'a> TyLoweringContext<'a> {
    pub fn new(db: &'a dyn HirDatabase, resolver: &'a Resolver) -> Self {
        let impl_trait_counter = std::cell::Cell::new(0);
        let impl_trait_mode = ImplTraitLoweringMode::Disallowed;
        let type_param_mode = TypeParamLoweringMode::Placeholder;
        let in_binders = DebruijnIndex::INNERMOST;
        let opaque_type_data = std::cell::RefCell::new(Vec::new());
        Self {
            db,
            resolver,
            in_binders,
            impl_trait_mode,
            impl_trait_counter,
            type_param_mode,
            opaque_type_data,
        }
    }

    pub fn with_debruijn<T>(
        &self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&TyLoweringContext) -> T,
    ) -> T {
        let opaque_ty_data_vec = self.opaque_type_data.replace(Vec::new());
        let new_ctx = Self {
            in_binders: debruijn,
            impl_trait_counter: std::cell::Cell::new(self.impl_trait_counter.get()),
            opaque_type_data: std::cell::RefCell::new(opaque_ty_data_vec),
            ..*self
        };
        let result = f(&new_ctx);
        self.impl_trait_counter.set(new_ctx.impl_trait_counter.get());
        self.opaque_type_data.replace(new_ctx.opaque_type_data.into_inner());
        result
    }

    pub fn with_shifted_in<T>(
        &self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&TyLoweringContext) -> T,
    ) -> T {
        self.with_debruijn(self.in_binders.shifted_in_from(debruijn), f)
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
            TypeRef::Never => Ty::Never,
            TypeRef::Tuple(inner) => {
                let inner_tys: Arc<[Ty]> = inner.iter().map(|tr| Ty::from_hir(ctx, tr)).collect();
                Ty::Tuple(inner_tys.len(), Substs(inner_tys))
            }
            TypeRef::Path(path) => {
                let (ty, res_) = Ty::from_hir_path(ctx, path);
                res = res_;
                ty
            }
            TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::Raw(lower_to_chalk_mutability(*mutability), Substs::single(inner_ty))
            }
            TypeRef::Array(inner) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::Array(Substs::single(inner_ty))
            }
            TypeRef::Slice(inner) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::Slice(Substs::single(inner_ty))
            }
            TypeRef::Reference(inner, _, mutability) => {
                let inner_ty = Ty::from_hir(ctx, inner);
                Ty::Ref(lower_to_chalk_mutability(*mutability), Substs::single(inner_ty))
            }
            TypeRef::Placeholder => Ty::Unknown,
            TypeRef::Fn(params, is_varargs) => {
                let substs = Substs(params.iter().map(|tr| Ty::from_hir(ctx, tr)).collect());
                Ty::Function(FnPointer {
                    num_args: substs.len() - 1,
                    sig: FnSig { variadic: *is_varargs },
                    substs,
                })
            }
            TypeRef::DynTrait(bounds) => {
                let self_ty = Ty::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0));
                let predicates = ctx.with_shifted_in(DebruijnIndex::ONE, |ctx| {
                    bounds
                        .iter()
                        .flat_map(|b| GenericPredicate::from_type_bound(ctx, b, self_ty.clone()))
                        .collect()
                });
                Ty::Dyn(predicates)
            }
            TypeRef::ImplTrait(bounds) => {
                match ctx.impl_trait_mode {
                    ImplTraitLoweringMode::Opaque => {
                        let idx = ctx.impl_trait_counter.get();
                        ctx.impl_trait_counter.set(idx + 1);

                        assert!(idx as usize == ctx.opaque_type_data.borrow().len());
                        // this dance is to make sure the data is in the right
                        // place even if we encounter more opaque types while
                        // lowering the bounds
                        ctx.opaque_type_data
                            .borrow_mut()
                            .push(ReturnTypeImplTrait { bounds: Binders::new(1, Vec::new()) });
                        // We don't want to lower the bounds inside the binders
                        // we're currently in, because they don't end up inside
                        // those binders. E.g. when we have `impl Trait<impl
                        // OtherTrait<T>>`, the `impl OtherTrait<T>` can't refer
                        // to the self parameter from `impl Trait`, and the
                        // bounds aren't actually stored nested within each
                        // other, but separately. So if the `T` refers to a type
                        // parameter of the outer function, it's just one binder
                        // away instead of two.
                        let actual_opaque_type_data = ctx
                            .with_debruijn(DebruijnIndex::INNERMOST, |ctx| {
                                ReturnTypeImplTrait::from_hir(ctx, &bounds)
                            });
                        ctx.opaque_type_data.borrow_mut()[idx as usize] = actual_opaque_type_data;

                        let func = match ctx.resolver.generic_def() {
                            Some(GenericDefId::FunctionId(f)) => f,
                            _ => panic!("opaque impl trait lowering in non-function"),
                        };
                        let impl_trait_id = OpaqueTyId::ReturnTypeImplTrait(func, idx);
                        let generics = generics(ctx.db.upcast(), func.into());
                        let parameters = Substs::bound_vars(&generics, ctx.in_binders);
                        Ty::Alias(AliasTy::Opaque(OpaqueTy {
                            opaque_ty_id: impl_trait_id,
                            parameters,
                        }))
                    }
                    ImplTraitLoweringMode::Param => {
                        let idx = ctx.impl_trait_counter.get();
                        // FIXME we're probably doing something wrong here
                        ctx.impl_trait_counter.set(idx + count_impl_traits(type_ref) as u16);
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
                        // FIXME we're probably doing something wrong here
                        ctx.impl_trait_counter.set(idx + count_impl_traits(type_ref) as u16);
                        let (parent_params, self_params, list_params, _impl_trait_params) =
                            if let Some(def) = ctx.resolver.generic_def() {
                                let generics = generics(ctx.db.upcast(), def);
                                generics.provenance_split()
                            } else {
                                (0, 0, 0, 0)
                            };
                        Ty::BoundVar(BoundVar::new(
                            ctx.in_binders,
                            idx as usize + parent_params + self_params + list_params,
                        ))
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
            (Ty::select_associated_type(ctx, res, segment), None)
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
        infer_args: bool,
    ) -> (Ty, Option<TypeNs>) {
        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                // if this is a bare dyn Trait, we'll directly put the required ^0 for the self type in there
                let self_ty = if remaining_segments.len() == 0 {
                    Some(Ty::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)))
                } else {
                    None
                };
                let trait_ref =
                    TraitRef::from_resolved_path(ctx, trait_, resolved_segment, self_ty);
                let ty = if remaining_segments.len() == 1 {
                    let segment = remaining_segments.first().unwrap();
                    let found = associated_type_by_name_including_super_traits(
                        ctx.db,
                        trait_ref,
                        &segment.name,
                    );
                    match found {
                        Some((super_trait_ref, associated_ty)) => {
                            // FIXME handle type parameters on the segment
                            Ty::Alias(AliasTy::Projection(ProjectionTy {
                                associated_ty,
                                parameters: super_trait_ref.substs,
                            }))
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
                        Ty::BoundVar(BoundVar::new(ctx.in_binders, idx))
                    }
                }
            }
            TypeNs::SelfType(impl_id) => {
                let generics = generics(ctx.db.upcast(), impl_id.into());
                let substs = match ctx.type_param_mode {
                    TypeParamLoweringMode::Placeholder => {
                        Substs::type_params_for_generics(&generics)
                    }
                    TypeParamLoweringMode::Variable => {
                        Substs::bound_vars(&generics, ctx.in_binders)
                    }
                };
                ctx.db.impl_self_ty(impl_id).subst(&substs)
            }
            TypeNs::AdtSelfType(adt) => {
                let generics = generics(ctx.db.upcast(), adt.into());
                let substs = match ctx.type_param_mode {
                    TypeParamLoweringMode::Placeholder => {
                        Substs::type_params_for_generics(&generics)
                    }
                    TypeParamLoweringMode::Variable => {
                        Substs::bound_vars(&generics, ctx.in_binders)
                    }
                };
                ctx.db.ty(adt.into()).subst(&substs)
            }

            TypeNs::AdtId(it) => {
                Ty::from_hir_path_inner(ctx, resolved_segment, it.into(), infer_args)
            }
            TypeNs::BuiltinType(it) => {
                Ty::from_hir_path_inner(ctx, resolved_segment, it.into(), infer_args)
            }
            TypeNs::TypeAliasId(it) => {
                Ty::from_hir_path_inner(ctx, resolved_segment, it.into(), infer_args)
            }
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
        Ty::from_partly_resolved_hir_path(
            ctx,
            resolution,
            resolved_segment,
            remaining_segments,
            false,
        )
    }

    fn select_associated_type(
        ctx: &TyLoweringContext<'_>,
        res: Option<TypeNs>,
        segment: PathSegment<'_>,
    ) -> Ty {
        if let Some(res) = res {
            let ty =
                associated_type_shorthand_candidates(ctx.db, res, move |name, t, associated_ty| {
                    if name == segment.name {
                        let substs = match ctx.type_param_mode {
                            TypeParamLoweringMode::Placeholder => {
                                // if we're lowering to placeholders, we have to put
                                // them in now
                                let s = Substs::type_params(
                                    ctx.db,
                                    ctx.resolver.generic_def().expect(
                                        "there should be generics if there's a generic param",
                                    ),
                                );
                                t.substs.clone().subst_bound_vars(&s)
                            }
                            TypeParamLoweringMode::Variable => t.substs.clone(),
                        };
                        // We need to shift in the bound vars, since
                        // associated_type_shorthand_candidates does not do that
                        let substs = substs.shift_bound_vars(ctx.in_binders);
                        // FIXME handle type parameters on the segment
                        return Some(Ty::Alias(AliasTy::Projection(ProjectionTy {
                            associated_ty,
                            parameters: substs,
                        })));
                    }

                    None
                });

            ty.unwrap_or(Ty::Unknown)
        } else {
            Ty::Unknown
        }
    }

    fn from_hir_path_inner(
        ctx: &TyLoweringContext<'_>,
        segment: PathSegment<'_>,
        typeable: TyDefId,
        infer_args: bool,
    ) -> Ty {
        let generic_def = match typeable {
            TyDefId::BuiltinType(_) => None,
            TyDefId::AdtId(it) => Some(it.into()),
            TyDefId::TypeAliasId(it) => Some(it.into()),
        };
        let substs = substs_from_path_segment(ctx, segment, generic_def, infer_args);
        ctx.db.ty(typeable).subst(&substs)
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
        infer_args: bool,
    ) -> Substs {
        let last = path.segments().last().expect("path should have at least one segment");
        let (segment, generic_def) = match resolved {
            ValueTyDefId::FunctionId(it) => (last, Some(it.into())),
            ValueTyDefId::StructId(it) => (last, Some(it.into())),
            ValueTyDefId::UnionId(it) => (last, Some(it.into())),
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
        substs_from_path_segment(ctx, segment, generic_def, infer_args)
    }
}

fn substs_from_path_segment(
    ctx: &TyLoweringContext<'_>,
    segment: PathSegment<'_>,
    def_generic: Option<GenericDefId>,
    infer_args: bool,
) -> Substs {
    let mut substs = Vec::new();
    let def_generics = def_generic.map(|def| generics(ctx.db.upcast(), def));

    let (parent_params, self_params, type_params, impl_trait_params) =
        def_generics.map_or((0, 0, 0, 0), |g| g.provenance_split());
    let total_len = parent_params + self_params + type_params + impl_trait_params;

    substs.extend(iter::repeat(Ty::Unknown).take(parent_params));

    let mut had_explicit_type_args = false;

    if let Some(generic_args) = &segment.args_and_bindings {
        if !generic_args.has_self_type {
            substs.extend(iter::repeat(Ty::Unknown).take(self_params));
        }
        let expected_num =
            if generic_args.has_self_type { self_params + type_params } else { type_params };
        let skip = if generic_args.has_self_type && self_params == 0 { 1 } else { 0 };
        // if args are provided, it should be all of them, but we can't rely on that
        for arg in generic_args
            .args
            .iter()
            .filter(|arg| matches!(arg, GenericArg::Type(_)))
            .skip(skip)
            .take(expected_num)
        {
            match arg {
                GenericArg::Type(type_ref) => {
                    had_explicit_type_args = true;
                    let ty = Ty::from_hir(ctx, type_ref);
                    substs.push(ty);
                }
                GenericArg::Lifetime(_) => {}
            }
        }
    }

    // handle defaults. In expression or pattern path segments without
    // explicitly specified type arguments, missing type arguments are inferred
    // (i.e. defaults aren't used).
    if !infer_args || had_explicit_type_args {
        if let Some(def_generic) = def_generic {
            let defaults = ctx.db.generic_defaults(def_generic);
            assert_eq!(total_len, defaults.len());

            for default_ty in defaults.iter().skip(substs.len()) {
                // each default can depend on the previous parameters
                let substs_so_far = Substs(substs.clone().into());
                substs.push(default_ty.clone().subst(&substs_so_far));
            }
        }
    }

    // add placeholders for args that were not provided
    // FIXME: emit diagnostics in contexts where this is not allowed
    for _ in substs.len()..total_len {
        substs.push(Ty::Unknown);
    }
    assert_eq!(substs.len(), total_len);

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
        substs_from_path_segment(ctx, segment, Some(resolved.into()), false)
    }
}

impl GenericPredicate {
    pub(crate) fn from_where_predicate<'a>(
        ctx: &'a TyLoweringContext<'a>,
        where_predicate: &'a WherePredicate,
    ) -> impl Iterator<Item = GenericPredicate> + 'a {
        match where_predicate {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let self_ty = match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => Ty::from_hir(ctx, type_ref),
                    WherePredicateTypeTarget::TypeParam(param_id) => {
                        let generic_def = ctx.resolver.generic_def().expect("generics in scope");
                        let generics = generics(ctx.db.upcast(), generic_def);
                        let param_id =
                            hir_def::TypeParamId { parent: generic_def, local_id: *param_id };
                        match ctx.type_param_mode {
                            TypeParamLoweringMode::Placeholder => Ty::Placeholder(param_id),
                            TypeParamLoweringMode::Variable => {
                                let idx = generics.param_idx(param_id).expect("matching generics");
                                Ty::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, idx))
                            }
                        }
                    }
                };
                GenericPredicate::from_type_bound(ctx, bound, self_ty)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            WherePredicate::Lifetime { .. } => vec![].into_iter(),
        }
    }

    pub(crate) fn from_type_bound<'a>(
        ctx: &'a TyLoweringContext<'a>,
        bound: &'a TypeBound,
        self_ty: Ty,
    ) -> impl Iterator<Item = GenericPredicate> + 'a {
        let mut bindings = None;
        let trait_ref = match bound {
            TypeBound::Path(path) => {
                bindings = TraitRef::from_path(ctx, path, Some(self_ty));
                Some(
                    bindings.clone().map_or(GenericPredicate::Error, GenericPredicate::Implemented),
                )
            }
            TypeBound::Lifetime(_) => None,
            TypeBound::Error => Some(GenericPredicate::Error),
        };
        trait_ref.into_iter().chain(
            bindings
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
        TypeBound::Error | TypeBound::Lifetime(_) => None,
    };
    last_segment
        .into_iter()
        .flat_map(|segment| segment.args_and_bindings.into_iter())
        .flat_map(|args_and_bindings| args_and_bindings.bindings.iter())
        .flat_map(move |binding| {
            let found = associated_type_by_name_including_super_traits(
                ctx.db,
                trait_ref.clone(),
                &binding.name,
            );
            let (super_trait_ref, associated_ty) = match found {
                None => return SmallVec::<[GenericPredicate; 1]>::new(),
                Some(t) => t,
            };
            let projection_ty = ProjectionTy { associated_ty, parameters: super_trait_ref.substs };
            let mut preds = SmallVec::with_capacity(
                binding.type_ref.as_ref().map_or(0, |_| 1) + binding.bounds.len(),
            );
            if let Some(type_ref) = &binding.type_ref {
                let ty = Ty::from_hir(ctx, type_ref);
                let projection_predicate =
                    ProjectionPredicate { projection_ty: projection_ty.clone(), ty };
                preds.push(GenericPredicate::Projection(projection_predicate));
            }
            for bound in &binding.bounds {
                preds.extend(GenericPredicate::from_type_bound(
                    ctx,
                    bound,
                    Ty::Alias(AliasTy::Projection(projection_ty.clone())),
                ));
            }
            preds
        })
}

impl ReturnTypeImplTrait {
    fn from_hir(ctx: &TyLoweringContext, bounds: &[TypeBound]) -> Self {
        cov_mark::hit!(lower_rpit);
        let self_ty = Ty::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0));
        let predicates = ctx.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            bounds
                .iter()
                .flat_map(|b| GenericPredicate::from_type_bound(ctx, b, self_ty.clone()))
                .collect()
        });
        ReturnTypeImplTrait { bounds: Binders::new(1, predicates) }
    }
}

fn count_impl_traits(type_ref: &TypeRef) -> usize {
    let mut count = 0;
    type_ref.walk(&mut |type_ref| {
        if matches!(type_ref, TypeRef::ImplTrait(_)) {
            count += 1;
        }
    });
    count
}

/// Build the signature of a callable item (function, struct or enum variant).
pub fn callable_item_sig(db: &dyn HirDatabase, def: CallableDefId) -> PolyFnSig {
    match def {
        CallableDefId::FunctionId(f) => fn_sig_for_fn(db, f),
        CallableDefId::StructId(s) => fn_sig_for_struct_constructor(db, s),
        CallableDefId::EnumVariantId(e) => fn_sig_for_enum_variant_constructor(db, e),
    }
}

pub fn associated_type_shorthand_candidates<R>(
    db: &dyn HirDatabase,
    res: TypeNs,
    mut cb: impl FnMut(&Name, &TraitRef, TypeAliasId) -> Option<R>,
) -> Option<R> {
    let traits_from_env: Vec<_> = match res {
        TypeNs::SelfType(impl_id) => match db.impl_trait(impl_id) {
            None => vec![],
            Some(trait_ref) => vec![trait_ref.value],
        },
        TypeNs::GenericParam(param_id) => {
            let predicates = db.generic_predicates_for_param(param_id);
            let mut traits_: Vec<_> = predicates
                .iter()
                .filter_map(|pred| match &pred.value {
                    GenericPredicate::Implemented(tr) => Some(tr.clone()),
                    _ => None,
                })
                .collect();
            // Handle `Self::Type` referring to own associated type in trait definitions
            if let GenericDefId::TraitId(trait_id) = param_id.parent {
                let generics = generics(db.upcast(), trait_id.into());
                if generics.params.types[param_id.local_id].provenance
                    == TypeParamProvenance::TraitSelf
                {
                    let trait_ref = TraitRef {
                        trait_: trait_id,
                        substs: Substs::bound_vars(&generics, DebruijnIndex::INNERMOST),
                    };
                    traits_.push(trait_ref);
                }
            }
            traits_
        }
        _ => vec![],
    };

    for t in traits_from_env.into_iter().flat_map(move |t| all_super_trait_refs(db, t)) {
        let data = db.trait_data(t.trait_);

        for (name, assoc_id) in &data.items {
            match assoc_id {
                AssocItemId::TypeAliasId(alias) => {
                    if let Some(result) = cb(name, &t, *alias) {
                        return Some(result);
                    }
                }
                AssocItemId::FunctionId(_) | AssocItemId::ConstId(_) => {}
            }
        }
    }

    None
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>> {
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
        .filter(|pred| match pred {
            WherePredicate::ForLifetime { target, .. }
            | WherePredicate::TypeBound { target, .. } => match target {
                WherePredicateTypeTarget::TypeRef(type_ref) => {
                    Ty::from_hir_only_param(&ctx, type_ref) == Some(param_id)
                }
                WherePredicateTypeTarget::TypeParam(local_id) => *local_id == param_id.local_id,
            },
            WherePredicate::Lifetime { .. } => false,
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
                cov_mark::hit!(trait_self_implements_self);
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
pub(crate) fn generic_defaults_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Arc<[Binders<Ty>]> {
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let generic_params = generics(db.upcast(), def);

    let defaults = generic_params
        .iter()
        .enumerate()
        .map(|(idx, (_, p))| {
            let mut ty = p.default.as_ref().map_or(Ty::Unknown, |t| Ty::from_hir(&ctx, t));

            // Each default can only refer to previous parameters.
            ty.walk_mut_binders(
                &mut |ty, binders| match ty {
                    Ty::BoundVar(BoundVar { debruijn, index }) if *debruijn == binders => {
                        if *index >= idx {
                            // type variable default referring to parameter coming
                            // after it. This is forbidden (FIXME: report
                            // diagnostic)
                            *ty = Ty::Unknown;
                        }
                    }
                    _ => {}
                },
                DebruijnIndex::INNERMOST,
            );

            Binders::new(idx, ty)
        })
        .collect();

    defaults
}

fn fn_sig_for_fn(db: &dyn HirDatabase, def: FunctionId) -> PolyFnSig {
    let data = db.function_data(def);
    let resolver = def.resolver(db.upcast());
    let ctx_params = TyLoweringContext::new(db, &resolver)
        .with_impl_trait_mode(ImplTraitLoweringMode::Variable)
        .with_type_param_mode(TypeParamLoweringMode::Variable);
    let params = data.params.iter().map(|tr| Ty::from_hir(&ctx_params, tr)).collect::<Vec<_>>();
    let ctx_ret = TyLoweringContext::new(db, &resolver)
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(TypeParamLoweringMode::Variable);
    let ret = Ty::from_hir(&ctx_ret, &data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let num_binders = generics.len();
    Binders::new(num_binders, CallableSig::from_params_and_return(params, ret, data.is_varargs))
}

/// Build the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn(db: &dyn HirDatabase, def: FunctionId) -> Binders<Ty> {
    let generics = generics(db.upcast(), def.into());
    let substs = Substs::bound_vars(&generics, DebruijnIndex::INNERMOST);
    Binders::new(substs.len(), Ty::FnDef(def.into(), substs))
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

fn fn_sig_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> PolyFnSig {
    let struct_data = db.struct_data(def);
    let fields = struct_data.variant_data.fields();
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    let params =
        fields.iter().map(|(_, field)| Ty::from_hir(&ctx, &field.type_ref)).collect::<Vec<_>>();
    let ret = type_for_adt(db, def.into());
    Binders::new(ret.num_binders, CallableSig::from_params_and_return(params, ret.value, false))
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> Binders<Ty> {
    let struct_data = db.struct_data(def);
    if let StructKind::Unit = struct_data.variant_data.kind() {
        return type_for_adt(db, def.into());
    }
    let generics = generics(db.upcast(), def.into());
    let substs = Substs::bound_vars(&generics, DebruijnIndex::INNERMOST);
    Binders::new(substs.len(), Ty::FnDef(def.into(), substs))
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
    Binders::new(ret.num_binders, CallableSig::from_params_and_return(params, ret.value, false))
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor(db: &dyn HirDatabase, def: EnumVariantId) -> Binders<Ty> {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id].variant_data;
    if let StructKind::Unit = var_data.kind() {
        return type_for_adt(db, def.parent.into());
    }
    let generics = generics(db.upcast(), def.parent.into());
    let substs = Substs::bound_vars(&generics, DebruijnIndex::INNERMOST);
    Binders::new(substs.len(), Ty::FnDef(def.into(), substs))
}

fn type_for_adt(db: &dyn HirDatabase, adt: AdtId) -> Binders<Ty> {
    let generics = generics(db.upcast(), adt.into());
    let substs = Substs::bound_vars(&generics, DebruijnIndex::INNERMOST);
    Binders::new(substs.len(), Ty::adt_ty(adt, substs))
}

fn type_for_type_alias(db: &dyn HirDatabase, t: TypeAliasId) -> Binders<Ty> {
    let generics = generics(db.upcast(), t.into());
    let resolver = t.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(TypeParamLoweringMode::Variable);
    if db.type_alias_data(t).is_extern {
        Binders::new(0, Ty::ForeignType(t))
    } else {
        let substs = Substs::bound_vars(&generics, DebruijnIndex::INNERMOST);
        let type_ref = &db.type_alias_data(t).type_ref;
        let inner = Ty::from_hir(&ctx, type_ref.as_ref().unwrap_or(&TypeRef::Error));
        Binders::new(substs.len(), inner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableDefId {
    FunctionId(FunctionId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
}
impl_from!(FunctionId, StructId, EnumVariantId for CallableDefId);

impl CallableDefId {
    pub fn krate(self, db: &dyn HirDatabase) -> CrateId {
        let db = db.upcast();
        match self {
            CallableDefId::FunctionId(f) => f.lookup(db).module(db),
            CallableDefId::StructId(s) => s.lookup(db).container.module(db),
            CallableDefId::EnumVariantId(e) => e.parent.lookup(db).container.module(db),
        }
        .krate()
    }
}

impl From<CallableDefId> for GenericDefId {
    fn from(def: CallableDefId) -> GenericDefId {
        match def {
            CallableDefId::FunctionId(f) => f.into(),
            CallableDefId::StructId(s) => s.into(),
            CallableDefId::EnumVariantId(e) => e.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyDefId {
    BuiltinType(BuiltinType),
    AdtId(AdtId),
    TypeAliasId(TypeAliasId),
}
impl_from!(BuiltinType, AdtId(StructId, EnumId, UnionId), TypeAliasId for TyDefId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueTyDefId {
    FunctionId(FunctionId),
    StructId(StructId),
    UnionId(UnionId),
    EnumVariantId(EnumVariantId),
    ConstId(ConstId),
    StaticId(StaticId),
}
impl_from!(FunctionId, StructId, UnionId, EnumVariantId, ConstId, StaticId for ValueTyDefId);

/// Build the declared type of an item. This depends on the namespace; e.g. for
/// `struct Foo(usize)`, we have two types: The type of the struct itself, and
/// the constructor function `(usize) -> Foo` which lives in the values
/// namespace.
pub(crate) fn ty_query(db: &dyn HirDatabase, def: TyDefId) -> Binders<Ty> {
    match def {
        TyDefId::BuiltinType(it) => Binders::new(0, Ty::builtin(it)),
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
        ValueTyDefId::UnionId(it) => type_for_adt(db, it.into()),
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

pub(crate) fn const_param_ty_query(db: &dyn HirDatabase, def: ConstParamId) -> Ty {
    let parent_data = db.generic_params(def.parent);
    let data = &parent_data.consts[def.local_id];
    let resolver = def.parent.resolver(db.upcast());
    let ctx = TyLoweringContext::new(db, &resolver);

    Ty::from_hir(&ctx, &data.ty)
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

pub(crate) fn return_type_impl_traits(
    db: &dyn HirDatabase,
    def: hir_def::FunctionId,
) -> Option<Arc<Binders<ReturnTypeImplTraits>>> {
    // FIXME unify with fn_sig_for_fn instead of doing lowering twice, maybe
    let data = db.function_data(def);
    let resolver = def.resolver(db.upcast());
    let ctx_ret = TyLoweringContext::new(db, &resolver)
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(TypeParamLoweringMode::Variable);
    let _ret = Ty::from_hir(&ctx_ret, &data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let num_binders = generics.len();
    let return_type_impl_traits =
        ReturnTypeImplTraits { impl_traits: ctx_ret.opaque_type_data.into_inner() };
    if return_type_impl_traits.impl_traits.is_empty() {
        None
    } else {
        Some(Arc::new(Binders::new(num_binders, return_type_impl_traits)))
    }
}

pub(crate) fn lower_to_chalk_mutability(m: hir_def::type_ref::Mutability) -> Mutability {
    match m {
        hir_def::type_ref::Mutability::Shared => Mutability::Not,
        hir_def::type_ref::Mutability::Mut => Mutability::Mut,
    }
}
