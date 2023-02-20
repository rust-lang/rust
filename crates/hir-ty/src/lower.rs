//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `TyLoweringContext::lower_ty`.
//!  - Building the type for an item: This happens through the `ty` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
use std::{
    cell::{Cell, RefCell, RefMut},
    iter,
    sync::Arc,
};

use base_db::CrateId;
use chalk_ir::{
    cast::Cast, fold::Shift, fold::TypeFoldable, interner::HasInterner, Mutability, Safety,
};

use hir_def::{
    adt::StructKind,
    body::{Expander, LowerCtx},
    builtin_type::BuiltinType,
    generics::{
        TypeOrConstParamData, TypeParamProvenance, WherePredicate, WherePredicateTypeTarget,
    },
    lang_item::{lang_attr, LangItem},
    path::{GenericArg, ModPath, Path, PathKind, PathSegment, PathSegments},
    resolver::{HasResolver, Resolver, TypeNs},
    type_ref::{
        ConstScalarOrPath, TraitBoundModifier, TraitRef as HirTraitRef, TypeBound, TypeRef,
    },
    AdtId, AssocItemId, ConstId, ConstParamId, EnumId, EnumVariantId, FunctionId, GenericDefId,
    HasModule, ImplId, ItemContainerId, LocalFieldId, Lookup, ModuleDefId, StaticId, StructId,
    TraitId, TypeAliasId, TypeOrConstParamId, TypeParamId, UnionId, VariantId,
};
use hir_expand::{name::Name, ExpandResult};
use intern::Interned;
use itertools::Either;
use la_arena::{Arena, ArenaMap};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use stdx::{impl_from, never};
use syntax::ast;

use crate::{
    all_super_traits,
    consteval::{intern_const_scalar, path_to_const, unknown_const, unknown_const_as_generic},
    db::HirDatabase,
    make_binders,
    mapping::{from_chalk_trait_id, ToChalk},
    static_lifetime, to_assoc_type_id, to_chalk_trait_id, to_placeholder_idx,
    utils::Generics,
    utils::{all_super_trait_refs, associated_type_by_name_including_super_traits, generics},
    AliasEq, AliasTy, Binders, BoundVar, CallableSig, Const, DebruijnIndex, DynTy, FnPointer,
    FnSig, FnSubst, GenericArgData, ImplTraitId, Interner, ParamKind, PolyFnSig, ProjectionTy,
    QuantifiedWhereClause, QuantifiedWhereClauses, ReturnTypeImplTrait, ReturnTypeImplTraits,
    Substitution, TraitEnvironment, TraitRef, TraitRefExt, Ty, TyBuilder, TyKind, WhereClause,
};

#[derive(Debug)]
enum ImplTraitLoweringState {
    /// When turning `impl Trait` into opaque types, we have to collect the
    /// bounds at the same time to get the IDs correct (without becoming too
    /// complicated). I don't like using interior mutability (as for the
    /// counter), but I've tried and failed to make the lifetimes work for
    /// passing around a `&mut TyLoweringContext`. The core problem is that
    /// we're grouping the mutable data (the counter and this field) together
    /// with the immutable context (the references to the DB and resolver).
    /// Splitting this up would be a possible fix.
    Opaque(RefCell<Arena<ReturnTypeImplTrait>>),
    Param(Cell<u16>),
    Variable(Cell<u16>),
    Disallowed,
}
impl ImplTraitLoweringState {
    fn new(impl_trait_mode: ImplTraitLoweringMode) -> ImplTraitLoweringState {
        match impl_trait_mode {
            ImplTraitLoweringMode::Opaque => Self::Opaque(RefCell::new(Arena::new())),
            ImplTraitLoweringMode::Param => Self::Param(Cell::new(0)),
            ImplTraitLoweringMode::Variable => Self::Variable(Cell::new(0)),
            ImplTraitLoweringMode::Disallowed => Self::Disallowed,
        }
    }

    fn take(&self) -> Self {
        match self {
            Self::Opaque(x) => Self::Opaque(RefCell::new(x.take())),
            Self::Param(x) => Self::Param(Cell::new(x.get())),
            Self::Variable(x) => Self::Variable(Cell::new(x.get())),
            Self::Disallowed => Self::Disallowed,
        }
    }

    fn swap(&self, impl_trait_mode: &Self) {
        match (self, impl_trait_mode) {
            (Self::Opaque(x), Self::Opaque(y)) => x.swap(y),
            (Self::Param(x), Self::Param(y)) => x.swap(y),
            (Self::Variable(x), Self::Variable(y)) => x.swap(y),
            (Self::Disallowed, Self::Disallowed) => (),
            _ => panic!("mismatched lowering mode"),
        }
    }
}

#[derive(Debug)]
pub struct TyLoweringContext<'a> {
    pub db: &'a dyn HirDatabase,
    pub resolver: &'a Resolver,
    in_binders: DebruijnIndex,
    /// Note: Conceptually, it's thinkable that we could be in a location where
    /// some type params should be represented as placeholders, and others
    /// should be converted to variables. I think in practice, this isn't
    /// possible currently, so this should be fine for now.
    pub type_param_mode: ParamLoweringMode,
    impl_trait_mode: ImplTraitLoweringState,
    expander: RefCell<Option<Expander>>,
    /// Tracks types with explicit `?Sized` bounds.
    pub(crate) unsized_types: RefCell<FxHashSet<Ty>>,
}

impl<'a> TyLoweringContext<'a> {
    pub fn new(db: &'a dyn HirDatabase, resolver: &'a Resolver) -> Self {
        let impl_trait_mode = ImplTraitLoweringState::Disallowed;
        let type_param_mode = ParamLoweringMode::Placeholder;
        let in_binders = DebruijnIndex::INNERMOST;
        Self {
            db,
            resolver,
            in_binders,
            impl_trait_mode,
            type_param_mode,
            expander: RefCell::new(None),
            unsized_types: RefCell::default(),
        }
    }

    pub fn with_debruijn<T>(
        &self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&TyLoweringContext<'_>) -> T,
    ) -> T {
        let impl_trait_mode = self.impl_trait_mode.take();
        let expander = self.expander.take();
        let unsized_types = self.unsized_types.take();
        let new_ctx = Self {
            in_binders: debruijn,
            impl_trait_mode,
            expander: RefCell::new(expander),
            unsized_types: RefCell::new(unsized_types),
            ..*self
        };
        let result = f(&new_ctx);
        self.impl_trait_mode.swap(&new_ctx.impl_trait_mode);
        self.expander.replace(new_ctx.expander.into_inner());
        self.unsized_types.replace(new_ctx.unsized_types.into_inner());
        result
    }

    pub fn with_shifted_in<T>(
        &self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&TyLoweringContext<'_>) -> T,
    ) -> T {
        self.with_debruijn(self.in_binders.shifted_in_from(debruijn), f)
    }

    pub fn with_impl_trait_mode(self, impl_trait_mode: ImplTraitLoweringMode) -> Self {
        Self { impl_trait_mode: ImplTraitLoweringState::new(impl_trait_mode), ..self }
    }

    pub fn with_type_param_mode(self, type_param_mode: ParamLoweringMode) -> Self {
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
pub enum ParamLoweringMode {
    Placeholder,
    Variable,
}

impl<'a> TyLoweringContext<'a> {
    pub fn lower_ty(&self, type_ref: &TypeRef) -> Ty {
        self.lower_ty_ext(type_ref).0
    }

    fn generics(&self) -> Generics {
        generics(
            self.db.upcast(),
            self.resolver
                .generic_def()
                .expect("there should be generics if there's a generic param"),
        )
    }

    pub fn lower_ty_ext(&self, type_ref: &TypeRef) -> (Ty, Option<TypeNs>) {
        let mut res = None;
        let ty = match type_ref {
            TypeRef::Never => TyKind::Never.intern(Interner),
            TypeRef::Tuple(inner) => {
                let inner_tys = inner.iter().map(|tr| self.lower_ty(tr));
                TyKind::Tuple(inner_tys.len(), Substitution::from_iter(Interner, inner_tys))
                    .intern(Interner)
            }
            TypeRef::Path(path) => {
                let (ty, res_) = self.lower_path(path);
                res = res_;
                ty
            }
            TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = self.lower_ty(inner);
                TyKind::Raw(lower_to_chalk_mutability(*mutability), inner_ty).intern(Interner)
            }
            TypeRef::Array(inner, len) => {
                let inner_ty = self.lower_ty(inner);
                let const_len = const_or_path_to_chalk(
                    self.db,
                    self.resolver,
                    TyBuilder::usize(),
                    len,
                    self.type_param_mode,
                    || self.generics(),
                    self.in_binders,
                );

                TyKind::Array(inner_ty, const_len).intern(Interner)
            }
            TypeRef::Slice(inner) => {
                let inner_ty = self.lower_ty(inner);
                TyKind::Slice(inner_ty).intern(Interner)
            }
            TypeRef::Reference(inner, _, mutability) => {
                let inner_ty = self.lower_ty(inner);
                let lifetime = static_lifetime();
                TyKind::Ref(lower_to_chalk_mutability(*mutability), lifetime, inner_ty)
                    .intern(Interner)
            }
            TypeRef::Placeholder => TyKind::Error.intern(Interner),
            &TypeRef::Fn(ref params, variadic, is_unsafe) => {
                let substs = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
                    Substitution::from_iter(Interner, params.iter().map(|(_, tr)| ctx.lower_ty(tr)))
                });
                TyKind::Function(FnPointer {
                    num_binders: 0, // FIXME lower `for<'a> fn()` correctly
                    sig: FnSig {
                        abi: (),
                        safety: if is_unsafe { Safety::Unsafe } else { Safety::Safe },
                        variadic,
                    },
                    substitution: FnSubst(substs),
                })
                .intern(Interner)
            }
            TypeRef::DynTrait(bounds) => self.lower_dyn_trait(bounds),
            TypeRef::ImplTrait(bounds) => {
                match &self.impl_trait_mode {
                    ImplTraitLoweringState::Opaque(opaque_type_data) => {
                        let func = match self.resolver.generic_def() {
                            Some(GenericDefId::FunctionId(f)) => f,
                            _ => panic!("opaque impl trait lowering in non-function"),
                        };

                        // this dance is to make sure the data is in the right
                        // place even if we encounter more opaque types while
                        // lowering the bounds
                        let idx = opaque_type_data.borrow_mut().alloc(ReturnTypeImplTrait {
                            bounds: crate::make_single_type_binders(Vec::new()),
                        });
                        // We don't want to lower the bounds inside the binders
                        // we're currently in, because they don't end up inside
                        // those binders. E.g. when we have `impl Trait<impl
                        // OtherTrait<T>>`, the `impl OtherTrait<T>` can't refer
                        // to the self parameter from `impl Trait`, and the
                        // bounds aren't actually stored nested within each
                        // other, but separately. So if the `T` refers to a type
                        // parameter of the outer function, it's just one binder
                        // away instead of two.
                        let actual_opaque_type_data = self
                            .with_debruijn(DebruijnIndex::INNERMOST, |ctx| {
                                ctx.lower_impl_trait(bounds, func)
                            });
                        opaque_type_data.borrow_mut()[idx] = actual_opaque_type_data;

                        let impl_trait_id = ImplTraitId::ReturnTypeImplTrait(func, idx);
                        let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
                        let generics = generics(self.db.upcast(), func.into());
                        let parameters = generics.bound_vars_subst(self.db, self.in_binders);
                        TyKind::OpaqueType(opaque_ty_id, parameters).intern(Interner)
                    }
                    ImplTraitLoweringState::Param(counter) => {
                        let idx = counter.get();
                        // FIXME we're probably doing something wrong here
                        counter.set(idx + count_impl_traits(type_ref) as u16);
                        if let Some(def) = self.resolver.generic_def() {
                            let generics = generics(self.db.upcast(), def);
                            let param = generics
                                .iter()
                                .filter(|(_, data)| {
                                    matches!(
                                        data,
                                        TypeOrConstParamData::TypeParamData(data)
                                        if data.provenance == TypeParamProvenance::ArgumentImplTrait
                                    )
                                })
                                .nth(idx as usize)
                                .map_or(TyKind::Error, |(id, _)| {
                                    TyKind::Placeholder(to_placeholder_idx(self.db, id))
                                });
                            param.intern(Interner)
                        } else {
                            TyKind::Error.intern(Interner)
                        }
                    }
                    ImplTraitLoweringState::Variable(counter) => {
                        let idx = counter.get();
                        // FIXME we're probably doing something wrong here
                        counter.set(idx + count_impl_traits(type_ref) as u16);
                        let (
                            _parent_params,
                            self_params,
                            list_params,
                            const_params,
                            _impl_trait_params,
                        ) = if let Some(def) = self.resolver.generic_def() {
                            let generics = generics(self.db.upcast(), def);
                            generics.provenance_split()
                        } else {
                            (0, 0, 0, 0, 0)
                        };
                        TyKind::BoundVar(BoundVar::new(
                            self.in_binders,
                            idx as usize + self_params + list_params + const_params,
                        ))
                        .intern(Interner)
                    }
                    ImplTraitLoweringState::Disallowed => {
                        // FIXME: report error
                        TyKind::Error.intern(Interner)
                    }
                }
            }
            TypeRef::Macro(macro_call) => {
                let (mut expander, recursion_start) = {
                    match RefMut::filter_map(self.expander.borrow_mut(), Option::as_mut) {
                        // There already is an expander here, this means we are already recursing
                        Ok(expander) => (expander, false),
                        // No expander was created yet, so we are at the start of the expansion recursion
                        // and therefore have to create an expander.
                        Err(expander) => (
                            RefMut::map(expander, |it| {
                                it.insert(Expander::new(
                                    self.db.upcast(),
                                    macro_call.file_id,
                                    self.resolver.module(),
                                ))
                            }),
                            true,
                        ),
                    }
                };
                let ty = {
                    let macro_call = macro_call.to_node(self.db.upcast());
                    match expander.enter_expand::<ast::Type>(self.db.upcast(), macro_call) {
                        Ok(ExpandResult { value: Some((mark, expanded)), .. }) => {
                            let ctx = LowerCtx::new(self.db.upcast(), expander.current_file_id());
                            let type_ref = TypeRef::from_ast(&ctx, expanded);

                            drop(expander);
                            let ty = self.lower_ty(&type_ref);

                            self.expander
                                .borrow_mut()
                                .as_mut()
                                .unwrap()
                                .exit(self.db.upcast(), mark);
                            Some(ty)
                        }
                        _ => {
                            drop(expander);
                            None
                        }
                    }
                };

                // drop the expander, resetting it to pre-recursion state
                if recursion_start {
                    *self.expander.borrow_mut() = None;
                }
                ty.unwrap_or_else(|| TyKind::Error.intern(Interner))
            }
            TypeRef::Error => TyKind::Error.intern(Interner),
        };
        (ty, res)
    }

    /// This is only for `generic_predicates_for_param`, where we can't just
    /// lower the self types of the predicates since that could lead to cycles.
    /// So we just check here if the `type_ref` resolves to a generic param, and which.
    fn lower_ty_only_param(&self, type_ref: &TypeRef) -> Option<TypeOrConstParamId> {
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
            match self.resolver.resolve_path_in_type_ns(self.db.upcast(), path.mod_path()) {
                Some((it, None)) => it,
                _ => return None,
            };
        match resolution {
            TypeNs::GenericParam(param_id) => Some(param_id.into()),
            _ => None,
        }
    }

    pub(crate) fn lower_ty_relative_path(
        &self,
        ty: Ty,
        // We need the original resolution to lower `Self::AssocTy` correctly
        res: Option<TypeNs>,
        remaining_segments: PathSegments<'_>,
    ) -> (Ty, Option<TypeNs>) {
        match remaining_segments.len() {
            0 => (ty, res),
            1 => {
                // resolve unselected assoc types
                let segment = remaining_segments.first().unwrap();
                (self.select_associated_type(res, segment), None)
            }
            _ => {
                // FIXME report error (ambiguous associated type)
                (TyKind::Error.intern(Interner), None)
            }
        }
    }

    pub(crate) fn lower_partly_resolved_path(
        &self,
        resolution: TypeNs,
        resolved_segment: PathSegment<'_>,
        remaining_segments: PathSegments<'_>,
        infer_args: bool,
    ) -> (Ty, Option<TypeNs>) {
        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                let ty = match remaining_segments.len() {
                    1 => {
                        let trait_ref =
                            self.lower_trait_ref_from_resolved_path(trait_, resolved_segment, None);
                        let segment = remaining_segments.first().unwrap();
                        let found = self
                            .db
                            .trait_data(trait_ref.hir_trait_id())
                            .associated_type_by_name(segment.name);

                        match found {
                            Some(associated_ty) => {
                                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                                // that method to optionally take parent `Substitution` as we already know them at
                                // this point (`trait_ref.substitution`).
                                let substitution = self.substs_from_path_segment(
                                    segment,
                                    Some(associated_ty.into()),
                                    false,
                                    None,
                                );
                                let len_self =
                                    generics(self.db.upcast(), associated_ty.into()).len_self();
                                let substitution = Substitution::from_iter(
                                    Interner,
                                    substitution
                                        .iter(Interner)
                                        .take(len_self)
                                        .chain(trait_ref.substitution.iter(Interner)),
                                );
                                TyKind::Alias(AliasTy::Projection(ProjectionTy {
                                    associated_ty_id: to_assoc_type_id(associated_ty),
                                    substitution,
                                }))
                                .intern(Interner)
                            }
                            None => {
                                // FIXME: report error (associated type not found)
                                TyKind::Error.intern(Interner)
                            }
                        }
                    }
                    0 => {
                        // Trait object type without dyn; this should be handled in upstream. See
                        // `lower_path()`.
                        stdx::never!("unexpected fully resolved trait path");
                        TyKind::Error.intern(Interner)
                    }
                    _ => {
                        // FIXME report error (ambiguous associated type)
                        TyKind::Error.intern(Interner)
                    }
                };
                return (ty, None);
            }
            TypeNs::GenericParam(param_id) => {
                let generics = generics(
                    self.db.upcast(),
                    self.resolver.generic_def().expect("generics in scope"),
                );
                match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        TyKind::Placeholder(to_placeholder_idx(self.db, param_id.into()))
                    }
                    ParamLoweringMode::Variable => {
                        let idx = match generics.param_idx(param_id.into()) {
                            None => {
                                never!("no matching generics");
                                return (TyKind::Error.intern(Interner), None);
                            }
                            Some(idx) => idx,
                        };

                        TyKind::BoundVar(BoundVar::new(self.in_binders, idx))
                    }
                }
                .intern(Interner)
            }
            TypeNs::SelfType(impl_id) => {
                let def =
                    self.resolver.generic_def().expect("impl should have generic param scope");
                let generics = generics(self.db.upcast(), def);

                match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // `def` can be either impl itself or item within, and we need impl itself
                        // now.
                        let generics = generics.parent_generics().unwrap_or(&generics);
                        let subst = generics.placeholder_subst(self.db);
                        self.db.impl_self_ty(impl_id).substitute(Interner, &subst)
                    }
                    ParamLoweringMode::Variable => {
                        let starting_from = match def {
                            GenericDefId::ImplId(_) => 0,
                            // `def` is an item within impl. We need to substitute `BoundVar`s but
                            // remember that they are for parent (i.e. impl) generic params so they
                            // come after our own params.
                            _ => generics.len_self(),
                        };
                        TyBuilder::impl_self_ty(self.db, impl_id)
                            .fill_with_bound_vars(self.in_binders, starting_from)
                            .build()
                    }
                }
            }
            TypeNs::AdtSelfType(adt) => {
                let generics = generics(self.db.upcast(), adt.into());
                let substs = match self.type_param_mode {
                    ParamLoweringMode::Placeholder => generics.placeholder_subst(self.db),
                    ParamLoweringMode::Variable => {
                        generics.bound_vars_subst(self.db, self.in_binders)
                    }
                };
                self.db.ty(adt.into()).substitute(Interner, &substs)
            }

            TypeNs::AdtId(it) => self.lower_path_inner(resolved_segment, it.into(), infer_args),
            TypeNs::BuiltinType(it) => {
                self.lower_path_inner(resolved_segment, it.into(), infer_args)
            }
            TypeNs::TypeAliasId(it) => {
                self.lower_path_inner(resolved_segment, it.into(), infer_args)
            }
            // FIXME: report error
            TypeNs::EnumVariantId(_) => return (TyKind::Error.intern(Interner), None),
        };
        self.lower_ty_relative_path(ty, Some(resolution), remaining_segments)
    }

    pub(crate) fn lower_path(&self, path: &Path) -> (Ty, Option<TypeNs>) {
        // Resolve the path (in type namespace)
        if let Some(type_ref) = path.type_anchor() {
            let (ty, res) = self.lower_ty_ext(type_ref);
            return self.lower_ty_relative_path(ty, res, path.segments());
        }

        let (resolution, remaining_index) =
            match self.resolver.resolve_path_in_type_ns(self.db.upcast(), path.mod_path()) {
                Some(it) => it,
                None => return (TyKind::Error.intern(Interner), None),
            };

        if matches!(resolution, TypeNs::TraitId(_)) && remaining_index.is_none() {
            // trait object type without dyn
            let bound = TypeBound::Path(path.clone(), TraitBoundModifier::None);
            let ty = self.lower_dyn_trait(&[Interned::new(bound)]);
            return (ty, None);
        }

        let (resolved_segment, remaining_segments) = match remaining_index {
            None => (
                path.segments().last().expect("resolved path has at least one element"),
                PathSegments::EMPTY,
            ),
            Some(i) => (path.segments().get(i - 1).unwrap(), path.segments().skip(i)),
        };
        self.lower_partly_resolved_path(resolution, resolved_segment, remaining_segments, false)
    }

    fn select_associated_type(&self, res: Option<TypeNs>, segment: PathSegment<'_>) -> Ty {
        let Some((def, res)) = self.resolver.generic_def().zip(res) else {
            return TyKind::Error.intern(Interner);
        };
        let ty = named_associated_type_shorthand_candidates(
            self.db,
            def,
            res,
            Some(segment.name.clone()),
            move |name, t, associated_ty| {
                if name != segment.name {
                    return None;
                }

                let parent_subst = t.substitution.clone();
                let parent_subst = match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // if we're lowering to placeholders, we have to put them in now.
                        let generics = generics(self.db.upcast(), def);
                        let s = generics.placeholder_subst(self.db);
                        s.apply(parent_subst, Interner)
                    }
                    ParamLoweringMode::Variable => {
                        // We need to shift in the bound vars, since
                        // `named_associated_type_shorthand_candidates` does not do that.
                        parent_subst.shifted_in_from(Interner, self.in_binders)
                    }
                };

                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                // that method to optionally take parent `Substitution` as we already know them at
                // this point (`t.substitution`).
                let substs = self.substs_from_path_segment(
                    segment.clone(),
                    Some(associated_ty.into()),
                    false,
                    None,
                );

                let len_self = generics(self.db.upcast(), associated_ty.into()).len_self();

                let substs = Substitution::from_iter(
                    Interner,
                    substs.iter(Interner).take(len_self).chain(parent_subst.iter(Interner)),
                );

                Some(
                    TyKind::Alias(AliasTy::Projection(ProjectionTy {
                        associated_ty_id: to_assoc_type_id(associated_ty),
                        substitution: substs,
                    }))
                    .intern(Interner),
                )
            },
        );

        ty.unwrap_or_else(|| TyKind::Error.intern(Interner))
    }

    fn lower_path_inner(
        &self,
        segment: PathSegment<'_>,
        typeable: TyDefId,
        infer_args: bool,
    ) -> Ty {
        let generic_def = match typeable {
            TyDefId::BuiltinType(_) => None,
            TyDefId::AdtId(it) => Some(it.into()),
            TyDefId::TypeAliasId(it) => Some(it.into()),
        };
        let substs = self.substs_from_path_segment(segment, generic_def, infer_args, None);
        self.db.ty(typeable).substitute(Interner, &substs)
    }

    /// Collect generic arguments from a path into a `Substs`. See also
    /// `create_substs_for_ast_path` and `def_to_ty` in rustc.
    pub(super) fn substs_from_path(
        &self,
        path: &Path,
        // Note that we don't call `db.value_type(resolved)` here,
        // `ValueTyDefId` is just a convenient way to pass generics and
        // special-case enum variants
        resolved: ValueTyDefId,
        infer_args: bool,
    ) -> Substitution {
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
                let penultimate = len.checked_sub(2).and_then(|idx| path.segments().get(idx));
                let segment = match penultimate {
                    Some(segment) if segment.args_and_bindings.is_some() => segment,
                    _ => last,
                };
                (segment, Some(var.parent.into()))
            }
        };
        self.substs_from_path_segment(segment, generic_def, infer_args, None)
    }

    fn substs_from_path_segment(
        &self,
        segment: PathSegment<'_>,
        def: Option<GenericDefId>,
        infer_args: bool,
        explicit_self_ty: Option<Ty>,
    ) -> Substitution {
        // Remember that the item's own generic args come before its parent's.
        let mut substs = Vec::new();
        let def = if let Some(d) = def {
            d
        } else {
            return Substitution::empty(Interner);
        };
        let def_generics = generics(self.db.upcast(), def);
        let (parent_params, self_params, type_params, const_params, impl_trait_params) =
            def_generics.provenance_split();
        let item_len = self_params + type_params + const_params + impl_trait_params;
        let total_len = parent_params + item_len;

        let ty_error = TyKind::Error.intern(Interner).cast(Interner);

        let mut def_generic_iter = def_generics.iter_id();

        let fill_self_params = || {
            for x in explicit_self_ty
                .into_iter()
                .map(|x| x.cast(Interner))
                .chain(iter::repeat(ty_error.clone()))
                .take(self_params)
            {
                if let Some(id) = def_generic_iter.next() {
                    assert!(id.is_left());
                    substs.push(x);
                }
            }
        };
        let mut had_explicit_args = false;

        if let Some(generic_args) = &segment.args_and_bindings {
            if !generic_args.has_self_type {
                fill_self_params();
            }
            let expected_num = if generic_args.has_self_type {
                self_params + type_params + const_params
            } else {
                type_params + const_params
            };
            let skip = if generic_args.has_self_type && self_params == 0 { 1 } else { 0 };
            // if args are provided, it should be all of them, but we can't rely on that
            for arg in generic_args
                .args
                .iter()
                .filter(|arg| !matches!(arg, GenericArg::Lifetime(_)))
                .skip(skip)
                .take(expected_num)
            {
                if let Some(id) = def_generic_iter.next() {
                    if let Some(x) = generic_arg_to_chalk(
                        self.db,
                        id,
                        arg,
                        &mut (),
                        |_, type_ref| self.lower_ty(type_ref),
                        |_, c, ty| {
                            const_or_path_to_chalk(
                                self.db,
                                self.resolver,
                                ty,
                                c,
                                self.type_param_mode,
                                || self.generics(),
                                self.in_binders,
                            )
                        },
                    ) {
                        had_explicit_args = true;
                        substs.push(x);
                    } else {
                        // we just filtered them out
                        never!("Unexpected lifetime argument");
                    }
                }
            }
        } else {
            fill_self_params();
        }

        // These params include those of parent.
        let remaining_params: SmallVec<[_; 2]> = def_generic_iter
            .map(|eid| match eid {
                Either::Left(_) => ty_error.clone(),
                Either::Right(x) => unknown_const_as_generic(self.db.const_param_ty(x)),
            })
            .collect();
        assert_eq!(remaining_params.len() + substs.len(), total_len);

        // handle defaults. In expression or pattern path segments without
        // explicitly specified type arguments, missing type arguments are inferred
        // (i.e. defaults aren't used).
        // Generic parameters for associated types are not supposed to have defaults, so we just
        // ignore them.
        let is_assoc_ty = if let GenericDefId::TypeAliasId(id) = def {
            let container = id.lookup(self.db.upcast()).container;
            matches!(container, ItemContainerId::TraitId(_))
        } else {
            false
        };
        if !is_assoc_ty && (!infer_args || had_explicit_args) {
            let defaults = self.db.generic_defaults(def);
            assert_eq!(total_len, defaults.len());
            let parent_from = item_len - substs.len();

            for (idx, default_ty) in defaults[substs.len()..item_len].iter().enumerate() {
                // each default can depend on the previous parameters
                let substs_so_far = Substitution::from_iter(
                    Interner,
                    substs.iter().cloned().chain(remaining_params[idx..].iter().cloned()),
                );
                substs.push(default_ty.clone().substitute(Interner, &substs_so_far));
            }

            // Keep parent's params as unknown.
            let mut remaining_params = remaining_params;
            substs.extend(remaining_params.drain(parent_from..));
        } else {
            substs.extend(remaining_params);
        }

        assert_eq!(substs.len(), total_len);
        Substitution::from_iter(Interner, substs)
    }

    fn lower_trait_ref_from_path(
        &self,
        path: &Path,
        explicit_self_ty: Option<Ty>,
    ) -> Option<TraitRef> {
        let resolved =
            match self.resolver.resolve_path_in_type_ns_fully(self.db.upcast(), path.mod_path())? {
                TypeNs::TraitId(tr) => tr,
                _ => return None,
            };
        let segment = path.segments().last().expect("path should have at least one segment");
        Some(self.lower_trait_ref_from_resolved_path(resolved, segment, explicit_self_ty))
    }

    pub(crate) fn lower_trait_ref_from_resolved_path(
        &self,
        resolved: TraitId,
        segment: PathSegment<'_>,
        explicit_self_ty: Option<Ty>,
    ) -> TraitRef {
        let substs = self.trait_ref_substs_from_path(segment, resolved, explicit_self_ty);
        TraitRef { trait_id: to_chalk_trait_id(resolved), substitution: substs }
    }

    fn lower_trait_ref(
        &self,
        trait_ref: &HirTraitRef,
        explicit_self_ty: Option<Ty>,
    ) -> Option<TraitRef> {
        self.lower_trait_ref_from_path(&trait_ref.path, explicit_self_ty)
    }

    fn trait_ref_substs_from_path(
        &self,
        segment: PathSegment<'_>,
        resolved: TraitId,
        explicit_self_ty: Option<Ty>,
    ) -> Substitution {
        self.substs_from_path_segment(segment, Some(resolved.into()), false, explicit_self_ty)
    }

    pub(crate) fn lower_where_predicate(
        &'a self,
        where_predicate: &'a WherePredicate,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + 'a {
        match where_predicate {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let self_ty = match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => self.lower_ty(type_ref),
                    WherePredicateTypeTarget::TypeOrConstParam(param_id) => {
                        let generic_def = self.resolver.generic_def().expect("generics in scope");
                        let generics = generics(self.db.upcast(), generic_def);
                        let param_id = hir_def::TypeOrConstParamId {
                            parent: generic_def,
                            local_id: *param_id,
                        };
                        let placeholder = to_placeholder_idx(self.db, param_id);
                        match self.type_param_mode {
                            ParamLoweringMode::Placeholder => TyKind::Placeholder(placeholder),
                            ParamLoweringMode::Variable => {
                                let idx = generics.param_idx(param_id).expect("matching generics");
                                TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, idx))
                            }
                        }
                        .intern(Interner)
                    }
                };
                self.lower_type_bound(bound, self_ty, ignore_bindings)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            WherePredicate::Lifetime { .. } => vec![].into_iter(),
        }
    }

    pub(crate) fn lower_type_bound(
        &'a self,
        bound: &'a TypeBound,
        self_ty: Ty,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + 'a {
        let mut bindings = None;
        let trait_ref = match bound {
            TypeBound::Path(path, TraitBoundModifier::None) => {
                bindings = self.lower_trait_ref_from_path(path, Some(self_ty));
                bindings
                    .clone()
                    .filter(|tr| {
                        // ignore `T: Drop` or `T: Destruct` bounds.
                        // - `T: ~const Drop` has a special meaning in Rust 1.61 that we don't implement.
                        //   (So ideally, we'd only ignore `~const Drop` here)
                        // - `Destruct` impls are built-in in 1.62 (current nightlies as of 08-04-2022), so until
                        //   the builtin impls are supported by Chalk, we ignore them here.
                        if let Some(lang) = lang_attr(self.db.upcast(), tr.hir_trait_id()) {
                            if lang == "drop" || lang == "destruct" {
                                return false;
                            }
                        }
                        true
                    })
                    .map(WhereClause::Implemented)
                    .map(crate::wrap_empty_binders)
            }
            TypeBound::Path(path, TraitBoundModifier::Maybe) => {
                let sized_trait = self
                    .db
                    .lang_item(self.resolver.krate(), LangItem::Sized)
                    .and_then(|lang_item| lang_item.as_trait());
                // Don't lower associated type bindings as the only possible relaxed trait bound
                // `?Sized` has no of them.
                // If we got another trait here ignore the bound completely.
                let trait_id = self
                    .lower_trait_ref_from_path(path, Some(self_ty.clone()))
                    .map(|trait_ref| trait_ref.hir_trait_id());
                if trait_id == sized_trait {
                    self.unsized_types.borrow_mut().insert(self_ty);
                }
                None
            }
            TypeBound::ForLifetime(_, path) => {
                // FIXME Don't silently drop the hrtb lifetimes here
                bindings = self.lower_trait_ref_from_path(path, Some(self_ty));
                bindings.clone().map(WhereClause::Implemented).map(crate::wrap_empty_binders)
            }
            TypeBound::Lifetime(_) => None,
            TypeBound::Error => None,
        };
        trait_ref.into_iter().chain(
            bindings
                .into_iter()
                .filter(move |_| !ignore_bindings)
                .flat_map(move |tr| self.assoc_type_bindings_from_type_bound(bound, tr)),
        )
    }

    fn assoc_type_bindings_from_type_bound(
        &'a self,
        bound: &'a TypeBound,
        trait_ref: TraitRef,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + 'a {
        let last_segment = match bound {
            TypeBound::Path(path, TraitBoundModifier::None) | TypeBound::ForLifetime(_, path) => {
                path.segments().last()
            }
            TypeBound::Path(_, TraitBoundModifier::Maybe)
            | TypeBound::Error
            | TypeBound::Lifetime(_) => None,
        };
        last_segment
            .into_iter()
            .filter_map(|segment| segment.args_and_bindings)
            .flat_map(|args_and_bindings| args_and_bindings.bindings.iter())
            .flat_map(move |binding| {
                let found = associated_type_by_name_including_super_traits(
                    self.db,
                    trait_ref.clone(),
                    &binding.name,
                );
                let (super_trait_ref, associated_ty) = match found {
                    None => return SmallVec::new(),
                    Some(t) => t,
                };
                // FIXME: `substs_from_path_segment()` pushes `TyKind::Error` for every parent
                // generic params. It's inefficient to splice the `Substitution`s, so we may want
                // that method to optionally take parent `Substitution` as we already know them at
                // this point (`super_trait_ref.substitution`).
                let substitution = self.substs_from_path_segment(
                    // FIXME: This is hack. We shouldn't really build `PathSegment` directly.
                    PathSegment { name: &binding.name, args_and_bindings: binding.args.as_deref() },
                    Some(associated_ty.into()),
                    false, // this is not relevant
                    Some(super_trait_ref.self_type_parameter(Interner)),
                );
                let self_params = generics(self.db.upcast(), associated_ty.into()).len_self();
                let substitution = Substitution::from_iter(
                    Interner,
                    substitution
                        .iter(Interner)
                        .take(self_params)
                        .chain(super_trait_ref.substitution.iter(Interner)),
                );
                let projection_ty = ProjectionTy {
                    associated_ty_id: to_assoc_type_id(associated_ty),
                    substitution,
                };
                let mut preds: SmallVec<[_; 1]> = SmallVec::with_capacity(
                    binding.type_ref.as_ref().map_or(0, |_| 1) + binding.bounds.len(),
                );
                if let Some(type_ref) = &binding.type_ref {
                    let ty = self.lower_ty(type_ref);
                    let alias_eq =
                        AliasEq { alias: AliasTy::Projection(projection_ty.clone()), ty };
                    preds.push(crate::wrap_empty_binders(WhereClause::AliasEq(alias_eq)));
                }
                for bound in binding.bounds.iter() {
                    preds.extend(self.lower_type_bound(
                        bound,
                        TyKind::Alias(AliasTy::Projection(projection_ty.clone())).intern(Interner),
                        false,
                    ));
                }
                preds
            })
    }

    fn lower_dyn_trait(&self, bounds: &[Interned<TypeBound>]) -> Ty {
        let self_ty = TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)).intern(Interner);
        // INVARIANT: The principal trait bound, if present, must come first. Others may be in any
        // order but should be in the same order for the same set but possibly different order of
        // bounds in the input.
        // INVARIANT: If this function returns `DynTy`, there should be at least one trait bound.
        // These invariants are utilized by `TyExt::dyn_trait()` and chalk.
        let bounds = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            let mut bounds: Vec<_> = bounds
                .iter()
                .flat_map(|b| ctx.lower_type_bound(b, self_ty.clone(), false))
                .collect();

            let mut multiple_regular_traits = false;
            let mut multiple_same_projection = false;
            bounds.sort_unstable_by(|lhs, rhs| {
                use std::cmp::Ordering;
                match (lhs.skip_binders(), rhs.skip_binders()) {
                    (WhereClause::Implemented(lhs), WhereClause::Implemented(rhs)) => {
                        let lhs_id = lhs.trait_id;
                        let lhs_is_auto = ctx.db.trait_data(from_chalk_trait_id(lhs_id)).is_auto;
                        let rhs_id = rhs.trait_id;
                        let rhs_is_auto = ctx.db.trait_data(from_chalk_trait_id(rhs_id)).is_auto;

                        if !lhs_is_auto && !rhs_is_auto {
                            multiple_regular_traits = true;
                        }
                        // Note that the ordering here is important; this ensures the invariant
                        // mentioned above.
                        (lhs_is_auto, lhs_id).cmp(&(rhs_is_auto, rhs_id))
                    }
                    (WhereClause::Implemented(_), _) => Ordering::Less,
                    (_, WhereClause::Implemented(_)) => Ordering::Greater,
                    (WhereClause::AliasEq(lhs), WhereClause::AliasEq(rhs)) => {
                        match (&lhs.alias, &rhs.alias) {
                            (AliasTy::Projection(lhs_proj), AliasTy::Projection(rhs_proj)) => {
                                // We only compare the `associated_ty_id`s. We shouldn't have
                                // multiple bounds for an associated type in the correct Rust code,
                                // and if we do, we error out.
                                if lhs_proj.associated_ty_id == rhs_proj.associated_ty_id {
                                    multiple_same_projection = true;
                                }
                                lhs_proj.associated_ty_id.cmp(&rhs_proj.associated_ty_id)
                            }
                            // We don't produce `AliasTy::Opaque`s yet.
                            _ => unreachable!(),
                        }
                    }
                    // We don't produce `WhereClause::{TypeOutlives, LifetimeOutlives}` yet.
                    _ => unreachable!(),
                }
            });

            if multiple_regular_traits || multiple_same_projection {
                return None;
            }

            if bounds.first().and_then(|b| b.trait_id()).is_none() {
                // When there's no trait bound, that's an error. This happens when the trait refs
                // are unresolved.
                return None;
            }

            // As multiple occurrences of the same auto traits *are* permitted, we dedulicate the
            // bounds. We shouldn't have repeated elements besides auto traits at this point.
            bounds.dedup();

            Some(QuantifiedWhereClauses::from_iter(Interner, bounds))
        });

        if let Some(bounds) = bounds {
            let bounds = crate::make_single_type_binders(bounds);
            TyKind::Dyn(DynTy { bounds, lifetime: static_lifetime() }).intern(Interner)
        } else {
            // FIXME: report error
            // (additional non-auto traits, associated type rebound, or no resolved trait)
            TyKind::Error.intern(Interner)
        }
    }

    fn lower_impl_trait(
        &self,
        bounds: &[Interned<TypeBound>],
        func: FunctionId,
    ) -> ReturnTypeImplTrait {
        cov_mark::hit!(lower_rpit);
        let self_ty = TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)).intern(Interner);
        let predicates = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            let mut predicates: Vec<_> = bounds
                .iter()
                .flat_map(|b| ctx.lower_type_bound(b, self_ty.clone(), false))
                .collect();

            if !ctx.unsized_types.borrow().contains(&self_ty) {
                let krate = func.lookup(ctx.db.upcast()).module(ctx.db.upcast()).krate();
                let sized_trait = ctx
                    .db
                    .lang_item(krate, LangItem::Sized)
                    .and_then(|lang_item| lang_item.as_trait().map(to_chalk_trait_id));
                let sized_clause = sized_trait.map(|trait_id| {
                    let clause = WhereClause::Implemented(TraitRef {
                        trait_id,
                        substitution: Substitution::from1(Interner, self_ty.clone()),
                    });
                    crate::wrap_empty_binders(clause)
                });
                predicates.extend(sized_clause.into_iter());
                predicates.shrink_to_fit();
            }
            predicates
        });
        ReturnTypeImplTrait { bounds: crate::make_single_type_binders(predicates) }
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
pub(crate) fn callable_item_sig(db: &dyn HirDatabase, def: CallableDefId) -> PolyFnSig {
    match def {
        CallableDefId::FunctionId(f) => fn_sig_for_fn(db, f),
        CallableDefId::StructId(s) => fn_sig_for_struct_constructor(db, s),
        CallableDefId::EnumVariantId(e) => fn_sig_for_enum_variant_constructor(db, e),
    }
}

pub fn associated_type_shorthand_candidates<R>(
    db: &dyn HirDatabase,
    def: GenericDefId,
    res: TypeNs,
    mut cb: impl FnMut(&Name, TypeAliasId) -> Option<R>,
) -> Option<R> {
    named_associated_type_shorthand_candidates(db, def, res, None, |name, _, id| cb(name, id))
}

fn named_associated_type_shorthand_candidates<R>(
    db: &dyn HirDatabase,
    // If the type parameter is defined in an impl and we're in a method, there
    // might be additional where clauses to consider
    def: GenericDefId,
    res: TypeNs,
    assoc_name: Option<Name>,
    // Do NOT let `cb` touch `TraitRef` outside of `TyLoweringContext`. Its substitution contains
    // free `BoundVar`s that need to be shifted and only `TyLoweringContext` knows how to do that
    // properly (see `TyLoweringContext::select_associated_type()`).
    mut cb: impl FnMut(&Name, &TraitRef, TypeAliasId) -> Option<R>,
) -> Option<R> {
    let mut search = |t| {
        all_super_trait_refs(db, t, |t| {
            let data = db.trait_data(t.hir_trait_id());

            for (name, assoc_id) in &data.items {
                if let AssocItemId::TypeAliasId(alias) = assoc_id {
                    if let Some(result) = cb(name, &t, *alias) {
                        return Some(result);
                    }
                }
            }
            None
        })
    };

    match res {
        TypeNs::SelfType(impl_id) => {
            // we're _in_ the impl -- the binders get added back later. Correct,
            // but it would be nice to make this more explicit
            let trait_ref = db.impl_trait(impl_id)?.into_value_and_skipped_binders().0;

            let impl_id_as_generic_def: GenericDefId = impl_id.into();
            if impl_id_as_generic_def != def {
                // `trait_ref` contains `BoundVar`s bound by impl's `Binders`, but here we need
                // `BoundVar`s from `def`'s point of view.
                // FIXME: A `HirDatabase` query may be handy if this process is needed in more
                // places. It'd be almost identical as `impl_trait_query` where `resolver` would be
                // of `def` instead of `impl_id`.
                let starting_idx = generics(db.upcast(), def).len_self();
                let subst = TyBuilder::subst_for_def(db, impl_id, None)
                    .fill_with_bound_vars(DebruijnIndex::INNERMOST, starting_idx)
                    .build();
                let trait_ref = subst.apply(trait_ref, Interner);
                search(trait_ref)
            } else {
                search(trait_ref)
            }
        }
        TypeNs::GenericParam(param_id) => {
            let predicates = db.generic_predicates_for_param(def, param_id.into(), assoc_name);
            let res = predicates.iter().find_map(|pred| match pred.skip_binders().skip_binders() {
                // FIXME: how to correctly handle higher-ranked bounds here?
                WhereClause::Implemented(tr) => search(
                    tr.clone()
                        .shifted_out_to(Interner, DebruijnIndex::ONE)
                        .expect("FIXME unexpected higher-ranked trait bound"),
                ),
                _ => None,
            });
            if let Some(_) = res {
                return res;
            }
            // Handle `Self::Type` referring to own associated type in trait definitions
            if let GenericDefId::TraitId(trait_id) = param_id.parent() {
                let trait_generics = generics(db.upcast(), trait_id.into());
                if trait_generics.params.type_or_consts[param_id.local_id()].is_trait_self() {
                    let def_generics = generics(db.upcast(), def);
                    let starting_idx = match def {
                        GenericDefId::TraitId(_) => 0,
                        // `def` is an item within trait. We need to substitute `BoundVar`s but
                        // remember that they are for parent (i.e. trait) generic params so they
                        // come after our own params.
                        _ => def_generics.len_self(),
                    };
                    let trait_ref = TyBuilder::trait_ref(db, trait_id)
                        .fill_with_bound_vars(DebruijnIndex::INNERMOST, starting_idx)
                        .build();
                    return search(trait_ref);
                }
            }
            None
        }
        _ => None,
    }
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>> {
    let var_data = variant_id.variant_data(db.upcast());
    let (resolver, def): (_, GenericDefId) = match variant_id {
        VariantId::StructId(it) => (it.resolver(db.upcast()), it.into()),
        VariantId::UnionId(it) => (it.resolver(db.upcast()), it.into()),
        VariantId::EnumVariantId(it) => (it.parent.resolver(db.upcast()), it.parent.into()),
    };
    let generics = generics(db.upcast(), def);
    let mut res = ArenaMap::default();
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, make_binders(db, &generics, ctx.lower_ty(&field_data.type_ref)));
    }
    Arc::new(res)
}

/// This query exists only to be used when resolving short-hand associated types
/// like `T::Item`.
///
/// See the analogous query in rustc and its comment:
/// <https://github.com/rust-lang/rust/blob/9150f844e2624eb013ec78ca08c1d416e6644026/src/librustc_typeck/astconv.rs#L46>
/// This is a query mostly to handle cycles somewhat gracefully; e.g. the
/// following bounds are disallowed: `T: Foo<U::Item>, U: Foo<T::Item>`, but
/// these are fine: `T: Foo<U::Item>, U: Foo<()>`.
pub(crate) fn generic_predicates_for_param_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
    param_id: TypeOrConstParamId,
    assoc_name: Option<Name>,
) -> Arc<[Binders<QuantifiedWhereClause>]> {
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    let generics = generics(db.upcast(), def);
    let mut predicates: Vec<_> = resolver
        .where_predicates_in_scope()
        // we have to filter out all other predicates *first*, before attempting to lower them
        .filter(|pred| match pred {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound, .. } => {
                match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => {
                        if ctx.lower_ty_only_param(type_ref) != Some(param_id) {
                            return false;
                        }
                    }
                    &WherePredicateTypeTarget::TypeOrConstParam(local_id) => {
                        let target_id = TypeOrConstParamId { parent: def, local_id };
                        if target_id != param_id {
                            return false;
                        }
                    }
                };

                match &**bound {
                    TypeBound::ForLifetime(_, path) | TypeBound::Path(path, _) => {
                        // Only lower the bound if the trait could possibly define the associated
                        // type we're looking for.

                        let assoc_name = match &assoc_name {
                            Some(it) => it,
                            None => return true,
                        };
                        let tr = match resolver
                            .resolve_path_in_type_ns_fully(db.upcast(), path.mod_path())
                        {
                            Some(TypeNs::TraitId(tr)) => tr,
                            _ => return false,
                        };

                        all_super_traits(db.upcast(), tr).iter().any(|tr| {
                            db.trait_data(*tr).items.iter().any(|(name, item)| {
                                matches!(item, AssocItemId::TypeAliasId(_)) && name == assoc_name
                            })
                        })
                    }
                    TypeBound::Lifetime(_) | TypeBound::Error => false,
                }
            }
            WherePredicate::Lifetime { .. } => false,
        })
        .flat_map(|pred| {
            ctx.lower_where_predicate(pred, true).map(|p| make_binders(db, &generics, p))
        })
        .collect();

    let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let explicitly_unsized_tys = ctx.unsized_types.into_inner();
    let implicitly_sized_predicates =
        implicitly_sized_clauses(db, param_id.parent, &explicitly_unsized_tys, &subst, &resolver)
            .map(|p| make_binders(db, &generics, crate::wrap_empty_binders(p)));
    predicates.extend(implicitly_sized_predicates);
    predicates.into()
}

pub(crate) fn generic_predicates_for_param_recover(
    _db: &dyn HirDatabase,
    _cycle: &[String],
    _def: &GenericDefId,
    _param_id: &TypeOrConstParamId,
    _assoc_name: &Option<Name>,
) -> Arc<[Binders<QuantifiedWhereClause>]> {
    Arc::new([])
}

pub(crate) fn trait_environment_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Arc<TraitEnvironment> {
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Placeholder);
    let mut traits_in_scope = Vec::new();
    let mut clauses = Vec::new();
    for pred in resolver.where_predicates_in_scope() {
        for pred in ctx.lower_where_predicate(pred, false) {
            if let WhereClause::Implemented(tr) = &pred.skip_binders() {
                traits_in_scope.push((tr.self_type_parameter(Interner).clone(), tr.hir_trait_id()));
            }
            let program_clause: chalk_ir::ProgramClause<Interner> = pred.cast(Interner);
            clauses.push(program_clause.into_from_env_clause(Interner));
        }
    }

    let container: Option<ItemContainerId> = match def {
        // FIXME: is there a function for this?
        GenericDefId::FunctionId(f) => Some(f.lookup(db.upcast()).container),
        GenericDefId::AdtId(_) => None,
        GenericDefId::TraitId(_) => None,
        GenericDefId::TypeAliasId(t) => Some(t.lookup(db.upcast()).container),
        GenericDefId::ImplId(_) => None,
        GenericDefId::EnumVariantId(_) => None,
        GenericDefId::ConstId(c) => Some(c.lookup(db.upcast()).container),
    };
    if let Some(ItemContainerId::TraitId(trait_id)) = container {
        // add `Self: Trait<T1, T2, ...>` to the environment in trait
        // function default implementations (and speculative code
        // inside consts or type aliases)
        cov_mark::hit!(trait_self_implements_self);
        let substs = TyBuilder::placeholder_subst(db, trait_id);
        let trait_ref = TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution: substs };
        let pred = WhereClause::Implemented(trait_ref);
        let program_clause: chalk_ir::ProgramClause<Interner> = pred.cast(Interner);
        clauses.push(program_clause.into_from_env_clause(Interner));
    }

    let subst = generics(db.upcast(), def).placeholder_subst(db);
    let explicitly_unsized_tys = ctx.unsized_types.into_inner();
    let implicitly_sized_clauses =
        implicitly_sized_clauses(db, def, &explicitly_unsized_tys, &subst, &resolver).map(|pred| {
            let program_clause: chalk_ir::ProgramClause<Interner> = pred.cast(Interner);
            program_clause.into_from_env_clause(Interner)
        });
    clauses.extend(implicitly_sized_clauses);

    let krate = def.module(db.upcast()).krate();

    let env = chalk_ir::Environment::new(Interner).add_clauses(Interner, clauses);

    Arc::new(TraitEnvironment { krate, traits_from_clauses: traits_in_scope, env })
}

/// Resolve the where clause(s) of an item with generics.
pub(crate) fn generic_predicates_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Arc<[Binders<QuantifiedWhereClause>]> {
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    let generics = generics(db.upcast(), def);

    let mut predicates = resolver
        .where_predicates_in_scope()
        .flat_map(|pred| {
            ctx.lower_where_predicate(pred, false).map(|p| make_binders(db, &generics, p))
        })
        .collect::<Vec<_>>();

    let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let explicitly_unsized_tys = ctx.unsized_types.into_inner();
    let implicitly_sized_predicates =
        implicitly_sized_clauses(db, def, &explicitly_unsized_tys, &subst, &resolver)
            .map(|p| make_binders(db, &generics, crate::wrap_empty_binders(p)));
    predicates.extend(implicitly_sized_predicates);
    predicates.into()
}

/// Generate implicit `: Sized` predicates for all generics that has no `?Sized` bound.
/// Exception is Self of a trait def.
fn implicitly_sized_clauses<'a>(
    db: &dyn HirDatabase,
    def: GenericDefId,
    explicitly_unsized_tys: &'a FxHashSet<Ty>,
    substitution: &'a Substitution,
    resolver: &Resolver,
) -> impl Iterator<Item = WhereClause> + 'a {
    let is_trait_def = matches!(def, GenericDefId::TraitId(..));
    let generic_args = &substitution.as_slice(Interner)[is_trait_def as usize..];
    let sized_trait = db
        .lang_item(resolver.krate(), LangItem::Sized)
        .and_then(|lang_item| lang_item.as_trait().map(to_chalk_trait_id));

    sized_trait.into_iter().flat_map(move |sized_trait| {
        let implicitly_sized_tys = generic_args
            .iter()
            .filter_map(|generic_arg| generic_arg.ty(Interner))
            .filter(move |&self_ty| !explicitly_unsized_tys.contains(self_ty));
        implicitly_sized_tys.map(move |self_ty| {
            WhereClause::Implemented(TraitRef {
                trait_id: sized_trait,
                substitution: Substitution::from1(Interner, self_ty.clone()),
            })
        })
    })
}

/// Resolve the default type params from generics
pub(crate) fn generic_defaults_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Arc<[Binders<chalk_ir::GenericArg<Interner>>]> {
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    let generic_params = generics(db.upcast(), def);
    let parent_start_idx = generic_params.len_self();

    let defaults = generic_params
        .iter()
        .enumerate()
        .map(|(idx, (id, p))| {
            let p = match p {
                TypeOrConstParamData::TypeParamData(p) => p,
                TypeOrConstParamData::ConstParamData(_) => {
                    // FIXME: implement const generic defaults
                    let val = unknown_const_as_generic(
                        db.const_param_ty(ConstParamId::from_unchecked(id)),
                    );
                    return make_binders(db, &generic_params, val);
                }
            };
            let mut ty =
                p.default.as_ref().map_or(TyKind::Error.intern(Interner), |t| ctx.lower_ty(t));

            // Each default can only refer to previous parameters.
            // Type variable default referring to parameter coming
            // after it is forbidden (FIXME: report diagnostic)
            ty = fallback_bound_vars(ty, idx, parent_start_idx);
            crate::make_binders(db, &generic_params, ty.cast(Interner))
        })
        .collect();

    defaults
}

pub(crate) fn generic_defaults_recover(
    db: &dyn HirDatabase,
    _cycle: &[String],
    def: &GenericDefId,
) -> Arc<[Binders<crate::GenericArg>]> {
    let generic_params = generics(db.upcast(), *def);
    // FIXME: this code is not covered in tests.
    // we still need one default per parameter
    let defaults = generic_params
        .iter_id()
        .map(|id| {
            let val = match id {
                itertools::Either::Left(_) => {
                    GenericArgData::Ty(TyKind::Error.intern(Interner)).intern(Interner)
                }
                itertools::Either::Right(id) => unknown_const_as_generic(db.const_param_ty(id)),
            };
            crate::make_binders(db, &generic_params, val)
        })
        .collect();

    defaults
}

fn fn_sig_for_fn(db: &dyn HirDatabase, def: FunctionId) -> PolyFnSig {
    let data = db.function_data(def);
    let resolver = def.resolver(db.upcast());
    let ctx_params = TyLoweringContext::new(db, &resolver)
        .with_impl_trait_mode(ImplTraitLoweringMode::Variable)
        .with_type_param_mode(ParamLoweringMode::Variable);
    let params = data.params.iter().map(|(_, tr)| ctx_params.lower_ty(tr)).collect::<Vec<_>>();
    let ctx_ret = TyLoweringContext::new(db, &resolver)
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(ParamLoweringMode::Variable);
    let ret = ctx_ret.lower_ty(&data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let sig = CallableSig::from_params_and_return(
        params,
        ret,
        data.is_varargs(),
        if data.has_unsafe_kw() { Safety::Unsafe } else { Safety::Safe },
    );
    make_binders(db, &generics, sig)
}

/// Build the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn(db: &dyn HirDatabase, def: FunctionId) -> Binders<Ty> {
    let generics = generics(db.upcast(), def.into());
    let substs = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    make_binders(
        db,
        &generics,
        TyKind::FnDef(CallableDefId::FunctionId(def).to_chalk(db), substs).intern(Interner),
    )
}

/// Build the declared type of a const.
fn type_for_const(db: &dyn HirDatabase, def: ConstId) -> Binders<Ty> {
    let data = db.const_data(def);
    let generics = generics(db.upcast(), def.into());
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);

    make_binders(db, &generics, ctx.lower_ty(&data.type_ref))
}

/// Build the declared type of a static.
fn type_for_static(db: &dyn HirDatabase, def: StaticId) -> Binders<Ty> {
    let data = db.static_data(def);
    let resolver = def.resolver(db.upcast());
    let ctx = TyLoweringContext::new(db, &resolver);

    Binders::empty(Interner, ctx.lower_ty(&data.type_ref))
}

fn fn_sig_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> PolyFnSig {
    let struct_data = db.struct_data(def);
    let fields = struct_data.variant_data.fields();
    let resolver = def.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    let params = fields.iter().map(|(_, field)| ctx.lower_ty(&field.type_ref)).collect::<Vec<_>>();
    let (ret, binders) = type_for_adt(db, def.into()).into_value_and_skipped_binders();
    Binders::new(binders, CallableSig::from_params_and_return(params, ret, false, Safety::Safe))
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> Binders<Ty> {
    let struct_data = db.struct_data(def);
    if let StructKind::Unit = struct_data.variant_data.kind() {
        return type_for_adt(db, def.into());
    }
    let generics = generics(db.upcast(), def.into());
    let substs = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    make_binders(
        db,
        &generics,
        TyKind::FnDef(CallableDefId::StructId(def).to_chalk(db), substs).intern(Interner),
    )
}

fn fn_sig_for_enum_variant_constructor(db: &dyn HirDatabase, def: EnumVariantId) -> PolyFnSig {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id];
    let fields = var_data.variant_data.fields();
    let resolver = def.parent.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    let params = fields.iter().map(|(_, field)| ctx.lower_ty(&field.type_ref)).collect::<Vec<_>>();
    let (ret, binders) = type_for_adt(db, def.parent.into()).into_value_and_skipped_binders();
    Binders::new(binders, CallableSig::from_params_and_return(params, ret, false, Safety::Safe))
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor(db: &dyn HirDatabase, def: EnumVariantId) -> Binders<Ty> {
    let enum_data = db.enum_data(def.parent);
    let var_data = &enum_data.variants[def.local_id].variant_data;
    if let StructKind::Unit = var_data.kind() {
        return type_for_adt(db, def.parent.into());
    }
    let generics = generics(db.upcast(), def.parent.into());
    let substs = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    make_binders(
        db,
        &generics,
        TyKind::FnDef(CallableDefId::EnumVariantId(def).to_chalk(db), substs).intern(Interner),
    )
}

fn type_for_adt(db: &dyn HirDatabase, adt: AdtId) -> Binders<Ty> {
    let generics = generics(db.upcast(), adt.into());
    let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let ty = TyKind::Adt(crate::AdtId(adt), subst).intern(Interner);
    make_binders(db, &generics, ty)
}

fn type_for_type_alias(db: &dyn HirDatabase, t: TypeAliasId) -> Binders<Ty> {
    let generics = generics(db.upcast(), t.into());
    let resolver = t.resolver(db.upcast());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    if db.type_alias_data(t).is_extern {
        Binders::empty(Interner, TyKind::Foreign(crate::to_foreign_def_id(t)).intern(Interner))
    } else {
        let type_ref = &db.type_alias_data(t).type_ref;
        let inner = ctx.lower_ty(type_ref.as_deref().unwrap_or(&TypeRef::Error));
        make_binders(db, &generics, inner)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableDefId {
    FunctionId(FunctionId),
    StructId(StructId),
    EnumVariantId(EnumVariantId),
}
impl_from!(FunctionId, StructId, EnumVariantId for CallableDefId);
impl From<CallableDefId> for ModuleDefId {
    fn from(def: CallableDefId) -> ModuleDefId {
        match def {
            CallableDefId::FunctionId(f) => ModuleDefId::FunctionId(f),
            CallableDefId::StructId(s) => ModuleDefId::AdtId(AdtId::StructId(s)),
            CallableDefId::EnumVariantId(e) => ModuleDefId::EnumVariantId(e),
        }
    }
}

impl CallableDefId {
    pub fn krate(self, db: &dyn HirDatabase) -> CrateId {
        let db = db.upcast();
        match self {
            CallableDefId::FunctionId(f) => f.lookup(db).module(db),
            CallableDefId::StructId(s) => s.lookup(db).container,
            CallableDefId::EnumVariantId(e) => e.parent.lookup(db).container,
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

impl ValueTyDefId {
    pub(crate) fn to_generic_def_id(self) -> Option<GenericDefId> {
        match self {
            Self::FunctionId(id) => Some(id.into()),
            Self::StructId(id) => Some(id.into()),
            Self::UnionId(id) => Some(id.into()),
            Self::EnumVariantId(var) => Some(var.into()),
            Self::ConstId(id) => Some(id.into()),
            Self::StaticId(_) => None,
        }
    }
}

/// Build the declared type of an item. This depends on the namespace; e.g. for
/// `struct Foo(usize)`, we have two types: The type of the struct itself, and
/// the constructor function `(usize) -> Foo` which lives in the values
/// namespace.
pub(crate) fn ty_query(db: &dyn HirDatabase, def: TyDefId) -> Binders<Ty> {
    match def {
        TyDefId::BuiltinType(it) => Binders::empty(Interner, TyBuilder::builtin(it)),
        TyDefId::AdtId(it) => type_for_adt(db, it),
        TyDefId::TypeAliasId(it) => type_for_type_alias(db, it),
    }
}

pub(crate) fn ty_recover(db: &dyn HirDatabase, _cycle: &[String], def: &TyDefId) -> Binders<Ty> {
    let generics = match *def {
        TyDefId::BuiltinType(_) => return Binders::empty(Interner, TyKind::Error.intern(Interner)),
        TyDefId::AdtId(it) => generics(db.upcast(), it.into()),
        TyDefId::TypeAliasId(it) => generics(db.upcast(), it.into()),
    };
    make_binders(db, &generics, TyKind::Error.intern(Interner))
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
    let impl_loc = impl_id.lookup(db.upcast());
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db.upcast());
    let _cx = stdx::panic_context::enter(format!(
        "impl_self_ty_query({impl_id:?} -> {impl_loc:?} -> {impl_data:?})"
    ));
    let generics = generics(db.upcast(), impl_id.into());
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    make_binders(db, &generics, ctx.lower_ty(&impl_data.self_ty))
}

// returns None if def is a type arg
pub(crate) fn const_param_ty_query(db: &dyn HirDatabase, def: ConstParamId) -> Ty {
    let parent_data = db.generic_params(def.parent());
    let data = &parent_data.type_or_consts[def.local_id()];
    let resolver = def.parent().resolver(db.upcast());
    let ctx = TyLoweringContext::new(db, &resolver);
    match data {
        TypeOrConstParamData::TypeParamData(_) => {
            never!();
            Ty::new(Interner, TyKind::Error)
        }
        TypeOrConstParamData::ConstParamData(d) => ctx.lower_ty(&d.ty),
    }
}

pub(crate) fn impl_self_ty_recover(
    db: &dyn HirDatabase,
    _cycle: &[String],
    impl_id: &ImplId,
) -> Binders<Ty> {
    let generics = generics(db.upcast(), (*impl_id).into());
    make_binders(db, &generics, TyKind::Error.intern(Interner))
}

pub(crate) fn impl_trait_query(db: &dyn HirDatabase, impl_id: ImplId) -> Option<Binders<TraitRef>> {
    let impl_loc = impl_id.lookup(db.upcast());
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db.upcast());
    let _cx = stdx::panic_context::enter(format!(
        "impl_trait_query({impl_id:?} -> {impl_loc:?} -> {impl_data:?})"
    ));
    let ctx =
        TyLoweringContext::new(db, &resolver).with_type_param_mode(ParamLoweringMode::Variable);
    let (self_ty, binders) = db.impl_self_ty(impl_id).into_value_and_skipped_binders();
    let target_trait = impl_data.target_trait.as_ref()?;
    Some(Binders::new(binders, ctx.lower_trait_ref(target_trait, Some(self_ty))?))
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
        .with_type_param_mode(ParamLoweringMode::Variable);
    let _ret = ctx_ret.lower_ty(&data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let return_type_impl_traits = ReturnTypeImplTraits {
        impl_traits: match ctx_ret.impl_trait_mode {
            ImplTraitLoweringState::Opaque(x) => x.into_inner(),
            _ => unreachable!(),
        },
    };
    if return_type_impl_traits.impl_traits.is_empty() {
        None
    } else {
        Some(Arc::new(make_binders(db, &generics, return_type_impl_traits)))
    }
}

pub(crate) fn lower_to_chalk_mutability(m: hir_def::type_ref::Mutability) -> Mutability {
    match m {
        hir_def::type_ref::Mutability::Shared => Mutability::Not,
        hir_def::type_ref::Mutability::Mut => Mutability::Mut,
    }
}

/// Checks if the provided generic arg matches its expected kind, then lower them via
/// provided closures. Use unknown if there was kind mismatch.
///
/// Returns `Some` of the lowered generic arg. `None` if the provided arg is a lifetime.
pub(crate) fn generic_arg_to_chalk<'a, T>(
    db: &dyn HirDatabase,
    kind_id: Either<TypeParamId, ConstParamId>,
    arg: &'a GenericArg,
    this: &mut T,
    for_type: impl FnOnce(&mut T, &TypeRef) -> Ty + 'a,
    for_const: impl FnOnce(&mut T, &ConstScalarOrPath, Ty) -> Const + 'a,
) -> Option<crate::GenericArg> {
    let kind = match kind_id {
        Either::Left(_) => ParamKind::Type,
        Either::Right(id) => {
            let ty = db.const_param_ty(id);
            ParamKind::Const(ty)
        }
    };
    Some(match (arg, kind) {
        (GenericArg::Type(type_ref), ParamKind::Type) => {
            let ty = for_type(this, type_ref);
            GenericArgData::Ty(ty).intern(Interner)
        }
        (GenericArg::Const(c), ParamKind::Const(c_ty)) => {
            GenericArgData::Const(for_const(this, c, c_ty)).intern(Interner)
        }
        (GenericArg::Const(_), ParamKind::Type) => {
            GenericArgData::Ty(TyKind::Error.intern(Interner)).intern(Interner)
        }
        (GenericArg::Type(t), ParamKind::Const(c_ty)) => {
            // We want to recover simple idents, which parser detects them
            // as types. Maybe here is not the best place to do it, but
            // it works.
            if let TypeRef::Path(p) = t {
                let p = p.mod_path();
                if p.kind == PathKind::Plain {
                    if let [n] = p.segments() {
                        let c = ConstScalarOrPath::Path(n.clone());
                        return Some(
                            GenericArgData::Const(for_const(this, &c, c_ty)).intern(Interner),
                        );
                    }
                }
            }
            unknown_const_as_generic(c_ty)
        }
        (GenericArg::Lifetime(_), _) => return None,
    })
}

pub(crate) fn const_or_path_to_chalk(
    db: &dyn HirDatabase,
    resolver: &Resolver,
    expected_ty: Ty,
    value: &ConstScalarOrPath,
    mode: ParamLoweringMode,
    args: impl FnOnce() -> Generics,
    debruijn: DebruijnIndex,
) -> Const {
    match value {
        ConstScalarOrPath::Scalar(s) => intern_const_scalar(*s, expected_ty),
        ConstScalarOrPath::Path(n) => {
            let path = ModPath::from_segments(PathKind::Plain, Some(n.clone()));
            path_to_const(db, resolver, &path, mode, args, debruijn)
                .unwrap_or_else(|| unknown_const(expected_ty))
        }
    }
}

/// Replaces any 'free' `BoundVar`s in `s` by `TyKind::Error` from the perspective of generic
/// parameter whose index is `param_index`. A `BoundVar` is free when it is or (syntactically)
/// appears after the generic parameter of `param_index`.
fn fallback_bound_vars<T: TypeFoldable<Interner> + HasInterner<Interner = Interner>>(
    s: T,
    param_index: usize,
    parent_start: usize,
) -> T {
    // Keep in mind that parent generic parameters, if any, come *after* those of the item in
    // question. In the diagrams below, `c*` and `p*` represent generic parameters of the item and
    // its parent respectively.
    let is_allowed = |index| {
        if param_index < parent_start {
            // The parameter of `param_index` is one from the item in question. Any parent generic
            // parameters or the item's generic parameters that come before `param_index` is
            // allowed.
            // [c1, .., cj, .., ck, p1, .., pl] where cj is `param_index`
            //  ^^^^^^              ^^^^^^^^^^ these are allowed
            !(param_index..parent_start).contains(&index)
        } else {
            // The parameter of `param_index` is one from the parent generics. Only parent generic
            // parameters that come before `param_index` are allowed.
            // [c1, .., ck, p1, .., pj, .., pl] where pj is `param_index`
            //              ^^^^^^ these are allowed
            (parent_start..param_index).contains(&index)
        }
    };

    crate::fold_free_vars(
        s,
        |bound, binders| {
            if bound.index_if_innermost().map_or(true, is_allowed) {
                bound.shifted_in_from(binders).to_ty(Interner)
            } else {
                TyKind::Error.intern(Interner)
            }
        },
        |ty, bound, binders| {
            if bound.index_if_innermost().map_or(true, is_allowed) {
                bound.shifted_in_from(binders).to_const(Interner, ty)
            } else {
                unknown_const(ty)
            }
        },
    )
}
