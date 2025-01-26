//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `TyLoweringContext::lower_ty`.
//!  - Building the type for an item: This happens through the `ty` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
pub(crate) mod diagnostics;

use std::{
    cell::OnceCell,
    iter, mem,
    ops::{self, Not as _},
};

use base_db::{ra_salsa::Cycle, CrateId};
use chalk_ir::{
    cast::Cast,
    fold::{Shift, TypeFoldable},
    interner::HasInterner,
    Mutability, Safety, TypeOutlives,
};

use either::Either;
use hir_def::{
    builtin_type::BuiltinType,
    data::{adt::StructKind, TraitFlags},
    expander::Expander,
    expr_store::HygieneId,
    generics::{
        GenericParamDataRef, TypeOrConstParamData, TypeParamProvenance, WherePredicate,
        WherePredicateTypeTarget,
    },
    lang_item::LangItem,
    nameres::MacroSubNs,
    path::{GenericArg, GenericArgs, ModPath, Path, PathKind, PathSegment, PathSegments},
    resolver::{HasResolver, LifetimeNs, ResolveValueResult, Resolver, TypeNs, ValueNs},
    type_ref::{
        ConstRef, LifetimeRef, PathId, TraitBoundModifier, TraitRef as HirTraitRef, TypeBound,
        TypeRef, TypeRefId, TypesMap, TypesSourceMap,
    },
    AdtId, AssocItemId, CallableDefId, ConstId, ConstParamId, DefWithBodyId, EnumId, EnumVariantId,
    FunctionId, GenericDefId, GenericParamId, HasModule, ImplId, InTypeConstLoc, ItemContainerId,
    LocalFieldId, Lookup, StaticId, StructId, TraitId, TypeAliasId, TypeOrConstParamId,
    TypeOwnerId, UnionId, VariantId,
};
use hir_expand::{name::Name, ExpandResult};
use la_arena::{Arena, ArenaMap};
use rustc_hash::FxHashSet;
use rustc_pattern_analysis::Captures;
use smallvec::SmallVec;
use stdx::{impl_from, never};
use syntax::ast;
use triomphe::{Arc, ThinArc};

use crate::{
    all_super_traits,
    consteval::{
        intern_const_ref, intern_const_scalar, path_to_const, unknown_const,
        unknown_const_as_generic,
    },
    db::HirDatabase,
    error_lifetime,
    generics::{generics, trait_self_param_idx, Generics},
    lower::diagnostics::*,
    make_binders,
    mapping::{from_chalk_trait_id, lt_to_placeholder_idx, ToChalk},
    static_lifetime, to_assoc_type_id, to_chalk_trait_id, to_placeholder_idx,
    utils::{
        all_super_trait_refs, associated_type_by_name_including_super_traits, InTypeConstIdMetadata,
    },
    AliasEq, AliasTy, Binders, BoundVar, CallableSig, Const, ConstScalar, DebruijnIndex, DynTy,
    FnAbi, FnPointer, FnSig, FnSubst, ImplTrait, ImplTraitId, ImplTraits, Interner, Lifetime,
    LifetimeData, LifetimeOutlives, ParamKind, PolyFnSig, ProgramClause, ProjectionTy,
    QuantifiedWhereClause, QuantifiedWhereClauses, Substitution, TraitEnvironment, TraitRef,
    TraitRefExt, Ty, TyBuilder, TyKind, WhereClause,
};

#[derive(Debug, Default)]
struct ImplTraitLoweringState {
    /// When turning `impl Trait` into opaque types, we have to collect the
    /// bounds at the same time to get the IDs correct (without becoming too
    /// complicated).
    mode: ImplTraitLoweringMode,
    // This is structured as a struct with fields and not as an enum because it helps with the borrow checker.
    opaque_type_data: Arena<ImplTrait>,
    param_and_variable_counter: u16,
}
impl ImplTraitLoweringState {
    fn new(mode: ImplTraitLoweringMode) -> ImplTraitLoweringState {
        Self { mode, opaque_type_data: Arena::new(), param_and_variable_counter: 0 }
    }
    fn param(counter: u16) -> Self {
        Self {
            mode: ImplTraitLoweringMode::Param,
            opaque_type_data: Arena::new(),
            param_and_variable_counter: counter,
        }
    }
    fn variable(counter: u16) -> Self {
        Self {
            mode: ImplTraitLoweringMode::Variable,
            opaque_type_data: Arena::new(),
            param_and_variable_counter: counter,
        }
    }
}

#[derive(Debug)]
pub struct TyLoweringContext<'a> {
    pub db: &'a dyn HirDatabase,
    resolver: &'a Resolver,
    generics: OnceCell<Option<Generics>>,
    types_map: &'a TypesMap,
    /// If this is set, that means we're in a context of a freshly expanded macro, and that means
    /// we should not use `TypeRefId` in diagnostics because the caller won't have the `TypesMap`,
    /// instead we need to put `TypeSource` from the source map.
    types_source_map: Option<&'a TypesSourceMap>,
    in_binders: DebruijnIndex,
    // FIXME: Should not be an `Option` but `Resolver` currently does not return owners in all cases
    // where expected
    owner: Option<TypeOwnerId>,
    /// Note: Conceptually, it's thinkable that we could be in a location where
    /// some type params should be represented as placeholders, and others
    /// should be converted to variables. I think in practice, this isn't
    /// possible currently, so this should be fine for now.
    pub type_param_mode: ParamLoweringMode,
    impl_trait_mode: ImplTraitLoweringState,
    expander: Option<Expander>,
    /// Tracks types with explicit `?Sized` bounds.
    pub(crate) unsized_types: FxHashSet<Ty>,
    pub(crate) diagnostics: Vec<TyLoweringDiagnostic>,
}

impl<'a> TyLoweringContext<'a> {
    pub fn new(
        db: &'a dyn HirDatabase,
        resolver: &'a Resolver,
        types_map: &'a TypesMap,
        owner: TypeOwnerId,
    ) -> Self {
        Self::new_maybe_unowned(db, resolver, types_map, None, Some(owner))
    }

    pub fn new_maybe_unowned(
        db: &'a dyn HirDatabase,
        resolver: &'a Resolver,
        types_map: &'a TypesMap,
        types_source_map: Option<&'a TypesSourceMap>,
        owner: Option<TypeOwnerId>,
    ) -> Self {
        let impl_trait_mode = ImplTraitLoweringState::new(ImplTraitLoweringMode::Disallowed);
        let type_param_mode = ParamLoweringMode::Placeholder;
        let in_binders = DebruijnIndex::INNERMOST;
        Self {
            db,
            resolver,
            generics: OnceCell::new(),
            types_map,
            types_source_map,
            owner,
            in_binders,
            impl_trait_mode,
            type_param_mode,
            expander: None,
            unsized_types: FxHashSet::default(),
            diagnostics: Vec::new(),
        }
    }

    pub fn with_debruijn<T>(
        &mut self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&mut TyLoweringContext<'_>) -> T,
    ) -> T {
        let old_debruijn = mem::replace(&mut self.in_binders, debruijn);
        let result = f(self);
        self.in_binders = old_debruijn;
        result
    }

    pub fn with_shifted_in<T>(
        &mut self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&mut TyLoweringContext<'_>) -> T,
    ) -> T {
        self.with_debruijn(self.in_binders.shifted_in_from(debruijn), f)
    }

    pub fn with_impl_trait_mode(self, impl_trait_mode: ImplTraitLoweringMode) -> Self {
        Self { impl_trait_mode: ImplTraitLoweringState::new(impl_trait_mode), ..self }
    }

    pub fn with_type_param_mode(self, type_param_mode: ParamLoweringMode) -> Self {
        Self { type_param_mode, ..self }
    }

    pub fn impl_trait_mode(&mut self, impl_trait_mode: ImplTraitLoweringMode) -> &mut Self {
        self.impl_trait_mode = ImplTraitLoweringState::new(impl_trait_mode);
        self
    }

    pub fn type_param_mode(&mut self, type_param_mode: ParamLoweringMode) -> &mut Self {
        self.type_param_mode = type_param_mode;
        self
    }

    pub fn push_diagnostic(&mut self, type_ref: TypeRefId, kind: TyLoweringDiagnosticKind) {
        let source = match self.types_source_map {
            Some(source_map) => {
                let Ok(source) = source_map.type_syntax(type_ref) else {
                    stdx::never!("error in synthetic type");
                    return;
                };
                Either::Right(source)
            }
            None => Either::Left(type_ref),
        };
        self.diagnostics.push(TyLoweringDiagnostic { source, kind });
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
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
    #[default]
    Disallowed,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParamLoweringMode {
    Placeholder,
    Variable,
}

impl<'a> TyLoweringContext<'a> {
    pub fn lower_ty(&mut self, type_ref: TypeRefId) -> Ty {
        self.lower_ty_ext(type_ref).0
    }

    pub fn lower_const(&mut self, const_ref: &ConstRef, const_type: Ty) -> Const {
        let Some(owner) = self.owner else { return unknown_const(const_type) };
        let debruijn = self.in_binders;
        const_or_path_to_chalk(
            self.db,
            self.resolver,
            owner,
            const_type,
            const_ref,
            self.type_param_mode,
            || self.generics(),
            debruijn,
        )
    }

    fn generics(&self) -> Option<&Generics> {
        self.generics
            .get_or_init(|| self.resolver.generic_def().map(|def| generics(self.db.upcast(), def)))
            .as_ref()
    }

    pub fn lower_ty_ext(&mut self, type_ref_id: TypeRefId) -> (Ty, Option<TypeNs>) {
        let mut res = None;
        let type_ref = &self.types_map[type_ref_id];
        let ty = match type_ref {
            TypeRef::Never => TyKind::Never.intern(Interner),
            TypeRef::Tuple(inner) => {
                let inner_tys = inner.iter().map(|&tr| self.lower_ty(tr));
                TyKind::Tuple(inner_tys.len(), Substitution::from_iter(Interner, inner_tys))
                    .intern(Interner)
            }
            TypeRef::Path(path) => {
                let (ty, res_) =
                    self.lower_path(path, PathId::from_type_ref_unchecked(type_ref_id));
                res = res_;
                ty
            }
            &TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = self.lower_ty(inner);
                TyKind::Raw(lower_to_chalk_mutability(mutability), inner_ty).intern(Interner)
            }
            TypeRef::Array(array) => {
                let inner_ty = self.lower_ty(array.ty);
                let const_len = self.lower_const(&array.len, TyBuilder::usize());
                TyKind::Array(inner_ty, const_len).intern(Interner)
            }
            &TypeRef::Slice(inner) => {
                let inner_ty = self.lower_ty(inner);
                TyKind::Slice(inner_ty).intern(Interner)
            }
            TypeRef::Reference(ref_) => {
                let inner_ty = self.lower_ty(ref_.ty);
                // FIXME: It should infer the eldided lifetimes instead of stubbing with static
                let lifetime = ref_
                    .lifetime
                    .as_ref()
                    .map_or_else(error_lifetime, |lr| self.lower_lifetime(lr));
                TyKind::Ref(lower_to_chalk_mutability(ref_.mutability), lifetime, inner_ty)
                    .intern(Interner)
            }
            TypeRef::Placeholder => TyKind::Error.intern(Interner),
            TypeRef::Fn(fn_) => {
                let substs = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
                    Substitution::from_iter(
                        Interner,
                        fn_.params().iter().map(|&(_, tr)| ctx.lower_ty(tr)),
                    )
                });
                TyKind::Function(FnPointer {
                    num_binders: 0, // FIXME lower `for<'a> fn()` correctly
                    sig: FnSig {
                        abi: fn_.abi().as_ref().map_or(FnAbi::Rust, FnAbi::from_symbol),
                        safety: if fn_.is_unsafe() { Safety::Unsafe } else { Safety::Safe },
                        variadic: fn_.is_varargs(),
                    },
                    substitution: FnSubst(substs),
                })
                .intern(Interner)
            }
            TypeRef::DynTrait(bounds) => self.lower_dyn_trait(bounds),
            TypeRef::ImplTrait(bounds) => {
                match self.impl_trait_mode.mode {
                    ImplTraitLoweringMode::Opaque => {
                        let origin = match self.resolver.generic_def() {
                            Some(GenericDefId::FunctionId(it)) => Either::Left(it),
                            Some(GenericDefId::TypeAliasId(it)) => Either::Right(it),
                            _ => panic!(
                                "opaque impl trait lowering must be in function or type alias"
                            ),
                        };

                        // this dance is to make sure the data is in the right
                        // place even if we encounter more opaque types while
                        // lowering the bounds
                        let idx = self.impl_trait_mode.opaque_type_data.alloc(ImplTrait {
                            bounds: crate::make_single_type_binders(Vec::default()),
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
                                ctx.lower_impl_trait(bounds, self.resolver.krate())
                            });
                        self.impl_trait_mode.opaque_type_data[idx] = actual_opaque_type_data;

                        let impl_trait_id = origin.either(
                            |f| ImplTraitId::ReturnTypeImplTrait(f, idx),
                            |a| ImplTraitId::TypeAliasImplTrait(a, idx),
                        );
                        let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
                        let generics =
                            generics(self.db.upcast(), origin.either(|f| f.into(), |a| a.into()));
                        let parameters = generics.bound_vars_subst(self.db, self.in_binders);
                        TyKind::OpaqueType(opaque_ty_id, parameters).intern(Interner)
                    }
                    ImplTraitLoweringMode::Param => {
                        let idx = self.impl_trait_mode.param_and_variable_counter;
                        // Count the number of `impl Trait` things that appear within our bounds.
                        // Since those have been emitted as implicit type args already.
                        self.impl_trait_mode.param_and_variable_counter =
                            idx + self.count_impl_traits(type_ref_id) as u16;
                        let db = self.db;
                        let kind = self
                            .generics()
                            .expect("param impl trait lowering must be in a generic def")
                            .iter()
                            .filter_map(|(id, data)| match (id, data) {
                                (
                                    GenericParamId::TypeParamId(id),
                                    GenericParamDataRef::TypeParamData(data),
                                ) if data.provenance == TypeParamProvenance::ArgumentImplTrait => {
                                    Some(id)
                                }
                                _ => None,
                            })
                            .nth(idx as usize)
                            .map_or(TyKind::Error, |id| {
                                TyKind::Placeholder(to_placeholder_idx(db, id.into()))
                            });
                        kind.intern(Interner)
                    }
                    ImplTraitLoweringMode::Variable => {
                        let idx = self.impl_trait_mode.param_and_variable_counter;
                        // Count the number of `impl Trait` things that appear within our bounds.
                        // Since t hose have been emitted as implicit type args already.
                        self.impl_trait_mode.param_and_variable_counter =
                            idx + self.count_impl_traits(type_ref_id) as u16;
                        let debruijn = self.in_binders;
                        let kind = self
                            .generics()
                            .expect("variable impl trait lowering must be in a generic def")
                            .iter()
                            .enumerate()
                            .filter_map(|(i, (id, data))| match (id, data) {
                                (
                                    GenericParamId::TypeParamId(_),
                                    GenericParamDataRef::TypeParamData(data),
                                ) if data.provenance == TypeParamProvenance::ArgumentImplTrait => {
                                    Some(i)
                                }
                                _ => None,
                            })
                            .nth(idx as usize)
                            .map_or(TyKind::Error, |id| {
                                TyKind::BoundVar(BoundVar { debruijn, index: id })
                            });
                        kind.intern(Interner)
                    }
                    ImplTraitLoweringMode::Disallowed => {
                        // FIXME: report error
                        TyKind::Error.intern(Interner)
                    }
                }
            }
            TypeRef::Macro(macro_call) => {
                let (expander, recursion_start) = {
                    match &mut self.expander {
                        // There already is an expander here, this means we are already recursing
                        Some(expander) => (expander, false),
                        // No expander was created yet, so we are at the start of the expansion recursion
                        // and therefore have to create an expander.
                        None => {
                            let expander = self.expander.insert(Expander::new(
                                self.db.upcast(),
                                macro_call.file_id,
                                self.resolver.module(),
                            ));
                            (expander, true)
                        }
                    }
                };
                let ty = {
                    let macro_call = macro_call.to_node(self.db.upcast());
                    let resolver = |path: &_| {
                        self.resolver
                            .resolve_path_as_macro(self.db.upcast(), path, Some(MacroSubNs::Bang))
                            .map(|(it, _)| it)
                    };
                    match expander.enter_expand::<ast::Type>(self.db.upcast(), macro_call, resolver)
                    {
                        Ok(ExpandResult { value: Some((mark, expanded)), .. }) => {
                            let (mut types_map, mut types_source_map) =
                                (TypesMap::default(), TypesSourceMap::default());

                            let mut ctx = expander.ctx(
                                self.db.upcast(),
                                &mut types_map,
                                &mut types_source_map,
                            );
                            // FIXME: Report syntax errors in expansion here
                            let type_ref = TypeRef::from_ast(&mut ctx, expanded.tree());

                            // Can't mutate `self`, must create a new instance, because of the lifetimes.
                            let mut inner_ctx = TyLoweringContext {
                                db: self.db,
                                resolver: self.resolver,
                                generics: self.generics.clone(),
                                types_map: &types_map,
                                types_source_map: Some(&types_source_map),
                                in_binders: self.in_binders,
                                owner: self.owner,
                                type_param_mode: self.type_param_mode,
                                impl_trait_mode: mem::take(&mut self.impl_trait_mode),
                                expander: self.expander.take(),
                                unsized_types: mem::take(&mut self.unsized_types),
                                diagnostics: mem::take(&mut self.diagnostics),
                            };

                            let ty = inner_ctx.lower_ty(type_ref);

                            self.impl_trait_mode = inner_ctx.impl_trait_mode;
                            self.expander = inner_ctx.expander;
                            self.unsized_types = inner_ctx.unsized_types;
                            self.diagnostics = inner_ctx.diagnostics;

                            self.expander.as_mut().unwrap().exit(mark);
                            Some(ty)
                        }
                        _ => None,
                    }
                };

                // drop the expander, resetting it to pre-recursion state
                if recursion_start {
                    self.expander = None;
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
    fn lower_ty_only_param(&mut self, type_ref_id: TypeRefId) -> Option<TypeOrConstParamId> {
        let type_ref = &self.types_map[type_ref_id];
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
        let resolution = match self
            .resolve_path_in_type_ns(path, &mut Self::on_path_diagnostic_callback(type_ref_id))
        {
            Some((it, None)) => it,
            _ => return None,
        };
        match resolution {
            TypeNs::GenericParam(param_id) => Some(param_id.into()),
            _ => None,
        }
    }

    pub(crate) fn lower_ty_relative_path(
        &mut self,
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
        &mut self,
        resolution: TypeNs,
        resolved_segment: PathSegment<'_>,
        remaining_segments: PathSegments<'_>,
        _resolved_segment_idx: u32,
        infer_args: bool,
        _on_diagnostic: &mut dyn FnMut(&mut Self, PathLoweringDiagnostic),
    ) -> (Ty, Option<TypeNs>) {
        let ty = match resolution {
            TypeNs::TraitId(trait_) => {
                let ty = match remaining_segments.len() {
                    1 => {
                        let trait_ref = self.lower_trait_ref_from_resolved_path(
                            trait_,
                            resolved_segment,
                            TyKind::Error.intern(Interner),
                        );
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
            TypeNs::TraitAliasId(_) => {
                // FIXME(trait_alias): Implement trait alias.
                return (TyKind::Error.intern(Interner), None);
            }
            TypeNs::GenericParam(param_id) => match self.type_param_mode {
                ParamLoweringMode::Placeholder => {
                    TyKind::Placeholder(to_placeholder_idx(self.db, param_id.into()))
                }
                ParamLoweringMode::Variable => {
                    let idx = match self
                        .generics()
                        .expect("generics in scope")
                        .type_or_const_param_idx(param_id.into())
                    {
                        None => {
                            never!("no matching generics");
                            return (TyKind::Error.intern(Interner), None);
                        }
                        Some(idx) => idx,
                    };

                    TyKind::BoundVar(BoundVar::new(self.in_binders, idx))
                }
            }
            .intern(Interner),
            TypeNs::SelfType(impl_id) => {
                let generics = self.generics().expect("impl should have generic param scope");

                match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // `def` can be either impl itself or item within, and we need impl itself
                        // now.
                        let generics = generics.parent_or_self();
                        let subst = generics.placeholder_subst(self.db);
                        self.db.impl_self_ty(impl_id).substitute(Interner, &subst)
                    }
                    ParamLoweringMode::Variable => {
                        let starting_from = match generics.def() {
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

    fn handle_type_ns_resolution(
        &mut self,
        resolution: &TypeNs,
        resolved_segment: PathSegment<'_>,
        resolved_segment_idx: usize,
        on_diagnostic: &mut dyn FnMut(&mut Self, PathLoweringDiagnostic),
    ) {
        let mut prohibit_generics_on_resolved = |reason| {
            if resolved_segment.args_and_bindings.is_some() {
                on_diagnostic(
                    self,
                    PathLoweringDiagnostic::GenericArgsProhibited {
                        segment: resolved_segment_idx as u32,
                        reason,
                    },
                );
            }
        };

        match resolution {
            TypeNs::SelfType(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::SelfTy)
            }
            TypeNs::GenericParam(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::TyParam)
            }
            TypeNs::AdtSelfType(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::SelfTy)
            }
            TypeNs::BuiltinType(_) => {
                prohibit_generics_on_resolved(GenericArgsProhibitedReason::PrimitiveTy)
            }
            TypeNs::AdtId(_)
            | TypeNs::EnumVariantId(_)
            | TypeNs::TypeAliasId(_)
            | TypeNs::TraitId(_)
            | TypeNs::TraitAliasId(_) => {}
        }
    }

    pub(crate) fn resolve_path_in_type_ns_fully(
        &mut self,
        path: &Path,
        on_diagnostic: &mut dyn FnMut(&mut Self, PathLoweringDiagnostic),
    ) -> Option<TypeNs> {
        let (res, unresolved) = self.resolve_path_in_type_ns(path, on_diagnostic)?;
        if unresolved.is_some() {
            return None;
        }
        Some(res)
    }

    pub(crate) fn resolve_path_in_type_ns(
        &mut self,
        path: &Path,
        on_diagnostic: &mut dyn FnMut(&mut Self, PathLoweringDiagnostic),
    ) -> Option<(TypeNs, Option<usize>)> {
        let (resolution, remaining_index, _, prefix_info) =
            self.resolver.resolve_path_in_type_ns_with_prefix_info(self.db.upcast(), path)?;
        let segments = path.segments();

        match path {
            // `segments.is_empty()` can occur with `self`.
            Path::Normal(..) if !segments.is_empty() => (),
            _ => return Some((resolution, remaining_index)),
        };

        let (module_segments, resolved_segment_idx, enum_segment) = match remaining_index {
            None if prefix_info.enum_variant => {
                (segments.strip_last_two(), segments.len() - 1, Some(segments.len() - 2))
            }
            None => (segments.strip_last(), segments.len() - 1, None),
            Some(i) => (segments.take(i - 1), i - 1, None),
        };

        for (i, mod_segment) in module_segments.iter().enumerate() {
            if mod_segment.args_and_bindings.is_some() {
                on_diagnostic(
                    self,
                    PathLoweringDiagnostic::GenericArgsProhibited {
                        segment: i as u32,
                        reason: GenericArgsProhibitedReason::Module,
                    },
                );
            }
        }

        if let Some(enum_segment) = enum_segment {
            if segments.get(enum_segment).is_some_and(|it| it.args_and_bindings.is_some())
                && segments.get(enum_segment + 1).is_some_and(|it| it.args_and_bindings.is_some())
            {
                on_diagnostic(
                    self,
                    PathLoweringDiagnostic::GenericArgsProhibited {
                        segment: (enum_segment + 1) as u32,
                        reason: GenericArgsProhibitedReason::EnumVariant,
                    },
                );
            }
        }

        self.handle_type_ns_resolution(
            &resolution,
            segments.get(resolved_segment_idx).expect("should have resolved segment"),
            resolved_segment_idx,
            on_diagnostic,
        );

        Some((resolution, remaining_index))
    }

    pub(crate) fn resolve_path_in_value_ns(
        &mut self,
        path: &Path,
        hygiene_id: HygieneId,
        on_diagnostic: &mut dyn FnMut(&mut Self, PathLoweringDiagnostic),
    ) -> Option<ResolveValueResult> {
        let (res, prefix_info) = self.resolver.resolve_path_in_value_ns_with_prefix_info(
            self.db.upcast(),
            path,
            hygiene_id,
        )?;

        let segments = path.segments();
        match path {
            // `segments.is_empty()` can occur with `self`.
            Path::Normal(..) if !segments.is_empty() => (),
            _ => return Some(res),
        };

        let (mod_segments, enum_segment) = match res {
            ResolveValueResult::Partial(_, unresolved_segment, _) => {
                (segments.take(unresolved_segment - 1), None)
            }
            ResolveValueResult::ValueNs(ValueNs::EnumVariantId(_), _)
                if prefix_info.enum_variant =>
            {
                (segments.strip_last_two(), segments.len().checked_sub(2))
            }
            ResolveValueResult::ValueNs(..) => (segments.strip_last(), None),
        };
        for (i, mod_segment) in mod_segments.iter().enumerate() {
            if mod_segment.args_and_bindings.is_some() {
                on_diagnostic(
                    self,
                    PathLoweringDiagnostic::GenericArgsProhibited {
                        segment: i as u32,
                        reason: GenericArgsProhibitedReason::Module,
                    },
                );
            }
        }

        if let Some(enum_segment) = enum_segment {
            if segments.get(enum_segment).is_some_and(|it| it.args_and_bindings.is_some())
                && segments.get(enum_segment + 1).is_some_and(|it| it.args_and_bindings.is_some())
            {
                on_diagnostic(
                    self,
                    PathLoweringDiagnostic::GenericArgsProhibited {
                        segment: (enum_segment + 1) as u32,
                        reason: GenericArgsProhibitedReason::EnumVariant,
                    },
                );
            }
        }

        match &res {
            ResolveValueResult::ValueNs(resolution, _) => {
                let resolved_segment_idx =
                    segments.len().checked_sub(1).unwrap_or_else(|| panic!("{path:?}"));
                let resolved_segment = segments.last().unwrap();

                let mut prohibit_generics_on_resolved = |reason| {
                    if resolved_segment.args_and_bindings.is_some() {
                        on_diagnostic(
                            self,
                            PathLoweringDiagnostic::GenericArgsProhibited {
                                segment: resolved_segment_idx as u32,
                                reason,
                            },
                        );
                    }
                };

                match resolution {
                    ValueNs::ImplSelf(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::SelfTy)
                    }
                    // FIXME: rustc generates E0107 (incorrect number of generic arguments) and not
                    // E0109 (generic arguments provided for a type that doesn't accept them) for
                    // consts and statics, presumably as a defense against future in which consts
                    // and statics can be generic, or just because it was easier for rustc implementors.
                    // That means we'll show the wrong error code. Because of us it's easier to do it
                    // this way :)
                    ValueNs::GenericParam(_) | ValueNs::ConstId(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::Const)
                    }
                    ValueNs::StaticId(_) => {
                        prohibit_generics_on_resolved(GenericArgsProhibitedReason::Static)
                    }
                    ValueNs::FunctionId(_) | ValueNs::StructId(_) | ValueNs::EnumVariantId(_) => {}
                    ValueNs::LocalBinding(_) => {}
                }
            }
            ResolveValueResult::Partial(resolution, unresolved_idx, _) => {
                let resolved_segment_idx = unresolved_idx - 1;
                let resolved_segment = segments.get(resolved_segment_idx).unwrap();
                self.handle_type_ns_resolution(
                    resolution,
                    resolved_segment,
                    resolved_segment_idx,
                    on_diagnostic,
                );
            }
        };
        Some(res)
    }

    fn on_path_diagnostic_callback(
        type_ref: TypeRefId,
    ) -> impl FnMut(&mut Self, PathLoweringDiagnostic) {
        move |this, diag| {
            this.push_diagnostic(type_ref, TyLoweringDiagnosticKind::PathDiagnostic(diag))
        }
    }

    pub(crate) fn lower_path(&mut self, path: &Path, path_id: PathId) -> (Ty, Option<TypeNs>) {
        // Resolve the path (in type namespace)
        if let Some(type_ref) = path.type_anchor() {
            let (ty, res) = self.lower_ty_ext(type_ref);
            return self.lower_ty_relative_path(ty, res, path.segments());
        }

        let (resolution, remaining_index) = match self.resolve_path_in_type_ns(
            path,
            &mut Self::on_path_diagnostic_callback(path_id.type_ref()),
        ) {
            Some(it) => it,
            None => return (TyKind::Error.intern(Interner), None),
        };

        if matches!(resolution, TypeNs::TraitId(_)) && remaining_index.is_none() {
            // trait object type without dyn
            let bound = TypeBound::Path(path_id, TraitBoundModifier::None);
            let ty = self.lower_dyn_trait(&[bound]);
            return (ty, None);
        }

        let (resolved_segment_idx, resolved_segment, remaining_segments) = match remaining_index {
            None => (
                path.segments().len() - 1,
                path.segments().last().expect("resolved path has at least one element"),
                PathSegments::EMPTY,
            ),
            Some(i) => (i - 1, path.segments().get(i - 1).unwrap(), path.segments().skip(i)),
        };

        self.lower_partly_resolved_path(
            resolution,
            resolved_segment,
            remaining_segments,
            resolved_segment_idx as u32,
            false,
            &mut Self::on_path_diagnostic_callback(path_id.type_ref()),
        )
    }

    fn select_associated_type(&mut self, res: Option<TypeNs>, segment: PathSegment<'_>) -> Ty {
        let Some((generics, res)) = self.generics().zip(res) else {
            return TyKind::Error.intern(Interner);
        };
        let ty = named_associated_type_shorthand_candidates(
            self.db,
            generics.def(),
            res,
            Some(segment.name.clone()),
            move |name, t, associated_ty| {
                let generics = self.generics().unwrap();

                if name != segment.name {
                    return None;
                }

                let parent_subst = t.substitution.clone();
                let parent_subst = match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        // if we're lowering to placeholders, we have to put them in now.
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
                let substs =
                    self.substs_from_path_segment(segment, Some(associated_ty.into()), false, None);

                let len_self =
                    crate::generics::generics(self.db.upcast(), associated_ty.into()).len_self();

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
        &mut self,
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
        &mut self,
        path: &Path,
        // Note that we don't call `db.value_type(resolved)` here,
        // `ValueTyDefId` is just a convenient way to pass generics and
        // special-case enum variants
        resolved: ValueTyDefId,
        infer_args: bool,
    ) -> Substitution {
        let last = path.segments().last();
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
                    Some(segment) if segment.args_and_bindings.is_some() => Some(segment),
                    _ => last,
                };
                (segment, Some(var.lookup(self.db.upcast()).parent.into()))
            }
        };
        if let Some(segment) = segment {
            self.substs_from_path_segment(segment, generic_def, infer_args, None)
        } else if let Some(generic_def) = generic_def {
            // lang item
            self.substs_from_args_and_bindings(None, Some(generic_def), infer_args, None)
        } else {
            Substitution::empty(Interner)
        }
    }

    pub(super) fn substs_from_path_segment(
        &mut self,
        segment: PathSegment<'_>,
        def: Option<GenericDefId>,
        infer_args: bool,
        explicit_self_ty: Option<Ty>,
    ) -> Substitution {
        self.substs_from_args_and_bindings(
            segment.args_and_bindings,
            def,
            infer_args,
            explicit_self_ty,
        )
    }

    fn substs_from_args_and_bindings(
        &mut self,
        args_and_bindings: Option<&GenericArgs>,
        def: Option<GenericDefId>,
        infer_args: bool,
        explicit_self_ty: Option<Ty>,
    ) -> Substitution {
        let Some(def) = def else { return Substitution::empty(Interner) };

        // Order is
        // - Optional Self parameter
        // - Lifetime parameters
        // - Type or Const parameters
        // - Parent parameters
        let def_generics = generics(self.db.upcast(), def);
        let (
            parent_params,
            self_param,
            type_params,
            const_params,
            impl_trait_params,
            lifetime_params,
        ) = def_generics.provenance_split();
        let item_len =
            self_param as usize + type_params + const_params + impl_trait_params + lifetime_params;
        let total_len = parent_params + item_len;

        let mut substs = Vec::new();

        // we need to iterate the lifetime and type/const params separately as our order of them
        // differs from the supplied syntax

        let ty_error = || TyKind::Error.intern(Interner).cast(Interner);
        let mut def_toc_iter = def_generics.iter_self_type_or_consts_id();
        let fill_self_param = || {
            if self_param {
                let self_ty = explicit_self_ty.map(|x| x.cast(Interner)).unwrap_or_else(ty_error);

                if let Some(id) = def_toc_iter.next() {
                    assert!(matches!(id, GenericParamId::TypeParamId(_)));
                    substs.push(self_ty);
                }
            }
        };
        let mut had_explicit_args = false;

        if let Some(&GenericArgs { ref args, has_self_type, .. }) = args_and_bindings {
            // Fill in the self param first
            if has_self_type && self_param {
                had_explicit_args = true;
                if let Some(id) = def_toc_iter.next() {
                    assert!(matches!(id, GenericParamId::TypeParamId(_)));
                    had_explicit_args = true;
                    if let GenericArg::Type(ty) = &args[0] {
                        substs.push(self.lower_ty(*ty).cast(Interner));
                    }
                }
            } else {
                fill_self_param()
            };

            // Then fill in the supplied lifetime args, or error lifetimes if there are too few
            // (default lifetimes aren't a thing)
            for arg in args
                .iter()
                .filter_map(|arg| match arg {
                    GenericArg::Lifetime(arg) => Some(self.lower_lifetime(arg)),
                    _ => None,
                })
                .chain(iter::repeat(error_lifetime()))
                .take(lifetime_params)
            {
                substs.push(arg.cast(Interner));
            }

            let skip = if has_self_type { 1 } else { 0 };
            // Fill in supplied type and const args
            // Note if non-lifetime args are provided, it should be all of them, but we can't rely on that
            for (arg, id) in args
                .iter()
                .filter(|arg| !matches!(arg, GenericArg::Lifetime(_)))
                .skip(skip)
                .take(type_params + const_params)
                .zip(def_toc_iter)
            {
                had_explicit_args = true;
                let arg = generic_arg_to_chalk(
                    self.db,
                    id,
                    arg,
                    self,
                    self.types_map,
                    |this, type_ref| this.lower_ty(type_ref),
                    |this, const_ref, ty| this.lower_const(const_ref, ty),
                    |this, lifetime_ref| this.lower_lifetime(lifetime_ref),
                );
                substs.push(arg);
            }
        } else {
            fill_self_param();
        }

        let param_to_err = |id| match id {
            GenericParamId::ConstParamId(x) => unknown_const_as_generic(self.db.const_param_ty(x)),
            GenericParamId::TypeParamId(_) => ty_error(),
            GenericParamId::LifetimeParamId(_) => error_lifetime().cast(Interner),
        };
        // handle defaults. In expression or pattern path segments without
        // explicitly specified type arguments, missing type arguments are inferred
        // (i.e. defaults aren't used).
        // Generic parameters for associated types are not supposed to have defaults, so we just
        // ignore them.
        let is_assoc_ty = || match def {
            GenericDefId::TypeAliasId(id) => {
                matches!(id.lookup(self.db.upcast()).container, ItemContainerId::TraitId(_))
            }
            _ => false,
        };
        let fill_defaults = (!infer_args || had_explicit_args) && !is_assoc_ty();
        if fill_defaults {
            let defaults = &*self.db.generic_defaults(def);
            let (item, _parent) = defaults.split_at(item_len);
            let parent_from = item_len - substs.len();

            let mut rem =
                def_generics.iter_id().skip(substs.len()).map(param_to_err).collect::<Vec<_>>();
            // Fill in defaults for type/const params
            for (idx, default_ty) in item[substs.len()..].iter().enumerate() {
                // each default can depend on the previous parameters
                let substs_so_far = Substitution::from_iter(
                    Interner,
                    substs.iter().cloned().chain(rem[idx..].iter().cloned()),
                );
                substs.push(default_ty.clone().substitute(Interner, &substs_so_far));
            }
            // Fill in remaining parent params
            substs.extend(rem.drain(parent_from..));
        } else {
            // Fill in remaining def params and parent params
            substs.extend(def_generics.iter_id().skip(substs.len()).map(param_to_err));
        }

        assert_eq!(substs.len(), total_len, "expected {} substs, got {}", total_len, substs.len());
        Substitution::from_iter(Interner, substs)
    }

    pub(crate) fn lower_trait_ref_from_resolved_path(
        &mut self,
        resolved: TraitId,
        segment: PathSegment<'_>,
        explicit_self_ty: Ty,
    ) -> TraitRef {
        let substs = self.trait_ref_substs_from_path(segment, resolved, explicit_self_ty);
        TraitRef { trait_id: to_chalk_trait_id(resolved), substitution: substs }
    }

    fn prohibit_generics(
        &mut self,
        path_id: PathId,
        idx: u32,
        segments: PathSegments<'_>,
        reason: GenericArgsProhibitedReason,
    ) {
        segments.iter().zip(idx..).for_each(|(segment, idx)| {
            if segment.args_and_bindings.is_some() {
                self.push_diagnostic(
                    path_id.type_ref(),
                    TyLoweringDiagnosticKind::PathDiagnostic(
                        PathLoweringDiagnostic::GenericArgsProhibited { segment: idx, reason },
                    ),
                );
            }
        });
    }

    fn lower_trait_ref_from_path(
        &mut self,
        path_id: PathId,
        explicit_self_ty: Ty,
    ) -> Option<TraitRef> {
        let path = &self.types_map[path_id];
        let resolved = match self.resolve_path_in_type_ns_fully(
            path,
            &mut Self::on_path_diagnostic_callback(path_id.type_ref()),
        )? {
            // FIXME(trait_alias): We need to handle trait alias here.
            TypeNs::TraitId(tr) => tr,
            _ => return None,
        };
        // Do this after we verify it's indeed a trait to not confuse the user if they're not modules.
        self.prohibit_generics(
            path_id,
            0,
            path.segments().strip_last(),
            GenericArgsProhibitedReason::Module,
        );
        let segment = path.segments().last().expect("path should have at least one segment");
        Some(self.lower_trait_ref_from_resolved_path(resolved, segment, explicit_self_ty))
    }

    fn lower_trait_ref(
        &mut self,
        trait_ref: &HirTraitRef,
        explicit_self_ty: Ty,
    ) -> Option<TraitRef> {
        self.lower_trait_ref_from_path(trait_ref.path, explicit_self_ty)
    }

    fn trait_ref_substs_from_path(
        &mut self,
        segment: PathSegment<'_>,
        resolved: TraitId,
        explicit_self_ty: Ty,
    ) -> Substitution {
        self.substs_from_path_segment(segment, Some(resolved.into()), false, Some(explicit_self_ty))
    }

    pub(crate) fn lower_where_predicate<'b>(
        &'b mut self,
        where_predicate: &'b WherePredicate,
        &def: &GenericDefId,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + use<'a, 'b> {
        match where_predicate {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let self_ty = match target {
                    WherePredicateTypeTarget::TypeRef(type_ref) => self.lower_ty(*type_ref),
                    &WherePredicateTypeTarget::TypeOrConstParam(local_id) => {
                        let param_id = hir_def::TypeOrConstParamId { parent: def, local_id };
                        match self.type_param_mode {
                            ParamLoweringMode::Placeholder => {
                                TyKind::Placeholder(to_placeholder_idx(self.db, param_id))
                            }
                            ParamLoweringMode::Variable => {
                                let idx = generics(self.db.upcast(), def)
                                    .type_or_const_param_idx(param_id)
                                    .expect("matching generics");
                                TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, idx))
                            }
                        }
                        .intern(Interner)
                    }
                };
                Either::Left(self.lower_type_bound(bound, self_ty, ignore_bindings))
            }
            WherePredicate::Lifetime { bound, target } => Either::Right(iter::once(
                crate::wrap_empty_binders(WhereClause::LifetimeOutlives(LifetimeOutlives {
                    a: self.lower_lifetime(bound),
                    b: self.lower_lifetime(target),
                })),
            )),
        }
        .into_iter()
    }

    pub(crate) fn lower_type_bound<'b>(
        &'b mut self,
        bound: &'b TypeBound,
        self_ty: Ty,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + use<'b, 'a> {
        let mut trait_ref = None;
        let clause = match bound {
            &TypeBound::Path(path, TraitBoundModifier::None) => {
                trait_ref = self.lower_trait_ref_from_path(path, self_ty);
                trait_ref.clone().map(WhereClause::Implemented).map(crate::wrap_empty_binders)
            }
            &TypeBound::Path(path, TraitBoundModifier::Maybe) => {
                let sized_trait = self
                    .db
                    .lang_item(self.resolver.krate(), LangItem::Sized)
                    .and_then(|lang_item| lang_item.as_trait());
                // Don't lower associated type bindings as the only possible relaxed trait bound
                // `?Sized` has no of them.
                // If we got another trait here ignore the bound completely.
                let trait_id = self
                    .lower_trait_ref_from_path(path, self_ty.clone())
                    .map(|trait_ref| trait_ref.hir_trait_id());
                if trait_id == sized_trait {
                    self.unsized_types.insert(self_ty);
                }
                None
            }
            &TypeBound::ForLifetime(_, path) => {
                // FIXME Don't silently drop the hrtb lifetimes here
                trait_ref = self.lower_trait_ref_from_path(path, self_ty);
                trait_ref.clone().map(WhereClause::Implemented).map(crate::wrap_empty_binders)
            }
            TypeBound::Lifetime(l) => {
                let lifetime = self.lower_lifetime(l);
                Some(crate::wrap_empty_binders(WhereClause::TypeOutlives(TypeOutlives {
                    ty: self_ty,
                    lifetime,
                })))
            }
            TypeBound::Use(_) | TypeBound::Error => None,
        };
        clause.into_iter().chain(
            trait_ref
                .filter(move |_| !ignore_bindings)
                .map(move |tr| self.assoc_type_bindings_from_type_bound(bound, tr))
                .into_iter()
                .flatten(),
        )
    }

    fn assoc_type_bindings_from_type_bound<'b>(
        &'b mut self,
        bound: &'b TypeBound,
        trait_ref: TraitRef,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + use<'b, 'a> {
        let last_segment = match bound {
            &TypeBound::Path(path, TraitBoundModifier::None) | &TypeBound::ForLifetime(_, path) => {
                self.types_map[path].segments().last()
            }
            TypeBound::Path(_, TraitBoundModifier::Maybe)
            | TypeBound::Use(_)
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
                    PathSegment { name: &binding.name, args_and_bindings: binding.args.as_ref() },
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
                let mut predicates: SmallVec<[_; 1]> = SmallVec::with_capacity(
                    binding.type_ref.as_ref().map_or(0, |_| 1) + binding.bounds.len(),
                );
                if let Some(type_ref) = binding.type_ref {
                    match (&self.types_map[type_ref], self.impl_trait_mode.mode) {
                        (TypeRef::ImplTrait(_), ImplTraitLoweringMode::Disallowed) => (),
                        (_, ImplTraitLoweringMode::Disallowed | ImplTraitLoweringMode::Opaque) => {
                            let ty = self.lower_ty(type_ref);
                            let alias_eq =
                                AliasEq { alias: AliasTy::Projection(projection_ty.clone()), ty };
                            predicates
                                .push(crate::wrap_empty_binders(WhereClause::AliasEq(alias_eq)));
                        }
                        (_, ImplTraitLoweringMode::Param | ImplTraitLoweringMode::Variable) => {
                            // Find the generic index for the target of our `bound`
                            let target_param_idx = self
                                .resolver
                                .where_predicates_in_scope()
                                .find_map(|(p, _)| match p {
                                    WherePredicate::TypeBound {
                                        target: WherePredicateTypeTarget::TypeOrConstParam(idx),
                                        bound: b,
                                    } if b == bound => Some(idx),
                                    _ => None,
                                });
                            let ty = if let Some(target_param_idx) = target_param_idx {
                                let mut counter = 0;
                                let generics = self.generics().expect("generics in scope");
                                for (idx, data) in generics.iter_self_type_or_consts() {
                                    // Count the number of `impl Trait` things that appear before
                                    // the target of our `bound`.
                                    // Our counter within `impl_trait_mode` should be that number
                                    // to properly lower each types within `type_ref`
                                    if data.type_param().is_some_and(|p| {
                                        p.provenance == TypeParamProvenance::ArgumentImplTrait
                                    }) {
                                        counter += 1;
                                    }
                                    if idx == *target_param_idx {
                                        break;
                                    }
                                }
                                let mut ext = TyLoweringContext::new_maybe_unowned(
                                    self.db,
                                    self.resolver,
                                    self.types_map,
                                    self.types_source_map,
                                    self.owner,
                                )
                                .with_type_param_mode(self.type_param_mode);
                                match self.impl_trait_mode.mode {
                                    ImplTraitLoweringMode::Param => {
                                        ext.impl_trait_mode =
                                            ImplTraitLoweringState::param(counter);
                                    }
                                    ImplTraitLoweringMode::Variable => {
                                        ext.impl_trait_mode =
                                            ImplTraitLoweringState::variable(counter);
                                    }
                                    _ => unreachable!(),
                                }
                                let ty = ext.lower_ty(type_ref);
                                self.diagnostics.extend(ext.diagnostics);
                                ty
                            } else {
                                self.lower_ty(type_ref)
                            };

                            let alias_eq =
                                AliasEq { alias: AliasTy::Projection(projection_ty.clone()), ty };
                            predicates
                                .push(crate::wrap_empty_binders(WhereClause::AliasEq(alias_eq)));
                        }
                    }
                }
                for bound in binding.bounds.iter() {
                    predicates.extend(self.lower_type_bound(
                        bound,
                        TyKind::Alias(AliasTy::Projection(projection_ty.clone())).intern(Interner),
                        false,
                    ));
                }
                predicates
            })
    }

    fn lower_dyn_trait(&mut self, bounds: &[TypeBound]) -> Ty {
        let self_ty = TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)).intern(Interner);
        // INVARIANT: The principal trait bound, if present, must come first. Others may be in any
        // order but should be in the same order for the same set but possibly different order of
        // bounds in the input.
        // INVARIANT: If this function returns `DynTy`, there should be at least one trait bound.
        // These invariants are utilized by `TyExt::dyn_trait()` and chalk.
        let mut lifetime = None;
        let bounds = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            let mut lowered_bounds = Vec::new();
            for b in bounds {
                ctx.lower_type_bound(b, self_ty.clone(), false).for_each(|b| {
                    let filter = match b.skip_binders() {
                        WhereClause::Implemented(_) | WhereClause::AliasEq(_) => true,
                        WhereClause::LifetimeOutlives(_) => false,
                        WhereClause::TypeOutlives(t) => {
                            lifetime = Some(t.lifetime.clone());
                            false
                        }
                    };
                    if filter {
                        lowered_bounds.push(b);
                    }
                });
            }

            let mut multiple_regular_traits = false;
            let mut multiple_same_projection = false;
            lowered_bounds.sort_unstable_by(|lhs, rhs| {
                use std::cmp::Ordering;
                match (lhs.skip_binders(), rhs.skip_binders()) {
                    (WhereClause::Implemented(lhs), WhereClause::Implemented(rhs)) => {
                        let lhs_id = lhs.trait_id;
                        let lhs_is_auto = ctx
                            .db
                            .trait_data(from_chalk_trait_id(lhs_id))
                            .flags
                            .contains(TraitFlags::IS_AUTO);
                        let rhs_id = rhs.trait_id;
                        let rhs_is_auto = ctx
                            .db
                            .trait_data(from_chalk_trait_id(rhs_id))
                            .flags
                            .contains(TraitFlags::IS_AUTO);

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
                    // `WhereClause::{TypeOutlives, LifetimeOutlives}` have been filtered out
                    _ => unreachable!(),
                }
            });

            if multiple_regular_traits || multiple_same_projection {
                return None;
            }

            lowered_bounds.first().and_then(|b| b.trait_id())?;

            // As multiple occurrences of the same auto traits *are* permitted, we deduplicate the
            // bounds. We shouldn't have repeated elements besides auto traits at this point.
            lowered_bounds.dedup();

            Some(QuantifiedWhereClauses::from_iter(Interner, lowered_bounds))
        });

        if let Some(bounds) = bounds {
            let bounds = crate::make_single_type_binders(bounds);
            TyKind::Dyn(DynTy {
                bounds,
                lifetime: match lifetime {
                    Some(it) => match it.bound_var(Interner) {
                        Some(bound_var) => bound_var
                            .shifted_out_to(DebruijnIndex::new(2))
                            .map(|bound_var| LifetimeData::BoundVar(bound_var).intern(Interner))
                            .unwrap_or(it),
                        None => it,
                    },
                    None => static_lifetime(),
                },
            })
            .intern(Interner)
        } else {
            // FIXME: report error
            // (additional non-auto traits, associated type rebound, or no resolved trait)
            TyKind::Error.intern(Interner)
        }
    }

    fn lower_impl_trait(&mut self, bounds: &[TypeBound], krate: CrateId) -> ImplTrait {
        cov_mark::hit!(lower_rpit);
        let self_ty = TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)).intern(Interner);
        let predicates = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            let mut predicates = Vec::new();
            for b in bounds {
                predicates.extend(ctx.lower_type_bound(b, self_ty.clone(), false));
            }

            if !ctx.unsized_types.contains(&self_ty) {
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
                predicates.extend(sized_clause);
            }
            predicates.shrink_to_fit();
            predicates
        });
        ImplTrait { bounds: crate::make_single_type_binders(predicates) }
    }

    pub fn lower_lifetime(&self, lifetime: &LifetimeRef) -> Lifetime {
        match self.resolver.resolve_lifetime(lifetime) {
            Some(resolution) => match resolution {
                LifetimeNs::Static => static_lifetime(),
                LifetimeNs::LifetimeParam(id) => match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        LifetimeData::Placeholder(lt_to_placeholder_idx(self.db, id))
                    }
                    ParamLoweringMode::Variable => {
                        let generics = self.generics().expect("generics in scope");
                        let idx = match generics.lifetime_idx(id) {
                            None => return error_lifetime(),
                            Some(idx) => idx,
                        };

                        LifetimeData::BoundVar(BoundVar::new(self.in_binders, idx))
                    }
                }
                .intern(Interner),
            },
            None => error_lifetime(),
        }
    }

    // FIXME: This does not handle macros!
    fn count_impl_traits(&self, type_ref: TypeRefId) -> usize {
        let mut count = 0;
        TypeRef::walk(type_ref, self.types_map, &mut |type_ref| {
            if matches!(type_ref, TypeRef::ImplTrait(_)) {
                count += 1;
            }
        });
        count
    }
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
            if res.is_some() {
                return res;
            }
            // Handle `Self::Type` referring to own associated type in trait definitions
            if let GenericDefId::TraitId(trait_id) = param_id.parent() {
                let trait_generics = generics(db.upcast(), trait_id.into());
                if trait_generics[param_id.local_id()].is_trait_self() {
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

pub(crate) type Diagnostics = Option<ThinArc<(), TyLoweringDiagnostic>>;

fn create_diagnostics(diagnostics: Vec<TyLoweringDiagnostic>) -> Diagnostics {
    (!diagnostics.is_empty()).then(|| ThinArc::from_header_and_iter((), diagnostics.into_iter()))
}

pub(crate) fn field_types_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>> {
    db.field_types_with_diagnostics(variant_id).0
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_with_diagnostics_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> (Arc<ArenaMap<LocalFieldId, Binders<Ty>>>, Diagnostics) {
    let var_data = variant_id.variant_data(db.upcast());
    let (resolver, def): (_, GenericDefId) = match variant_id {
        VariantId::StructId(it) => (it.resolver(db.upcast()), it.into()),
        VariantId::UnionId(it) => (it.resolver(db.upcast()), it.into()),
        VariantId::EnumVariantId(it) => {
            (it.resolver(db.upcast()), it.lookup(db.upcast()).parent.into())
        }
    };
    let generics = generics(db.upcast(), def);
    let mut res = ArenaMap::default();
    let mut ctx = TyLoweringContext::new(db, &resolver, var_data.types_map(), def.into())
        .with_type_param_mode(ParamLoweringMode::Variable);
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, make_binders(db, &generics, ctx.lower_ty(field_data.type_ref)));
    }
    (Arc::new(res), create_diagnostics(ctx.diagnostics))
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
) -> GenericPredicates {
    let resolver = def.resolver(db.upcast());
    let mut ctx = if let GenericDefId::FunctionId(_) = def {
        TyLoweringContext::new(db, &resolver, TypesMap::EMPTY, def.into())
            .with_impl_trait_mode(ImplTraitLoweringMode::Variable)
            .with_type_param_mode(ParamLoweringMode::Variable)
    } else {
        TyLoweringContext::new(db, &resolver, TypesMap::EMPTY, def.into())
            .with_type_param_mode(ParamLoweringMode::Variable)
    };
    let generics = generics(db.upcast(), def);

    // we have to filter out all other predicates *first*, before attempting to lower them
    let predicate = |pred: &_, def: &_, ctx: &mut TyLoweringContext<'_>| match pred {
        WherePredicate::ForLifetime { target, bound, .. }
        | WherePredicate::TypeBound { target, bound, .. } => {
            let invalid_target = match target {
                WherePredicateTypeTarget::TypeRef(type_ref) => {
                    ctx.lower_ty_only_param(*type_ref) != Some(param_id)
                }
                &WherePredicateTypeTarget::TypeOrConstParam(local_id) => {
                    let target_id = TypeOrConstParamId { parent: *def, local_id };
                    target_id != param_id
                }
            };
            if invalid_target {
                // If this is filtered out without lowering, `?Sized` is not gathered into `ctx.unsized_types`
                if let TypeBound::Path(_, TraitBoundModifier::Maybe) = bound {
                    ctx.lower_where_predicate(pred, def, true).for_each(drop);
                }
                return false;
            }

            match bound {
                &TypeBound::ForLifetime(_, path) | &TypeBound::Path(path, _) => {
                    // Only lower the bound if the trait could possibly define the associated
                    // type we're looking for.
                    let path = &ctx.types_map[path];

                    let Some(assoc_name) = &assoc_name else { return true };
                    let Some(TypeNs::TraitId(tr)) =
                        resolver.resolve_path_in_type_ns_fully(db.upcast(), path)
                    else {
                        return false;
                    };

                    all_super_traits(db.upcast(), tr).iter().any(|tr| {
                        db.trait_data(*tr).items.iter().any(|(name, item)| {
                            matches!(item, AssocItemId::TypeAliasId(_)) && name == assoc_name
                        })
                    })
                }
                TypeBound::Use(_) | TypeBound::Lifetime(_) | TypeBound::Error => false,
            }
        }
        WherePredicate::Lifetime { .. } => false,
    };
    let mut predicates = Vec::new();
    for (params, def) in resolver.all_generic_params() {
        ctx.types_map = &params.types_map;
        for pred in params.where_predicates() {
            if predicate(pred, def, &mut ctx) {
                predicates.extend(
                    ctx.lower_where_predicate(pred, def, true)
                        .map(|p| make_binders(db, &generics, p)),
                );
            }
        }
    }

    let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    if !subst.is_empty(Interner) {
        let explicitly_unsized_tys = ctx.unsized_types;
        if let Some(implicitly_sized_predicates) = implicitly_sized_clauses(
            db,
            param_id.parent,
            &explicitly_unsized_tys,
            &subst,
            &resolver,
        ) {
            predicates.extend(
                implicitly_sized_predicates
                    .map(|p| make_binders(db, &generics, crate::wrap_empty_binders(p))),
            );
        };
    }
    GenericPredicates(predicates.is_empty().not().then(|| predicates.into()))
}

pub(crate) fn generic_predicates_for_param_recover(
    _db: &dyn HirDatabase,
    _cycle: &Cycle,
    _def: &GenericDefId,
    _param_id: &TypeOrConstParamId,
    _assoc_name: &Option<Name>,
) -> GenericPredicates {
    GenericPredicates(None)
}

pub(crate) fn trait_environment_for_body_query(
    db: &dyn HirDatabase,
    def: DefWithBodyId,
) -> Arc<TraitEnvironment> {
    let Some(def) = def.as_generic_def_id(db.upcast()) else {
        let krate = def.module(db.upcast()).krate();
        return TraitEnvironment::empty(krate);
    };
    db.trait_environment(def)
}

pub(crate) fn trait_environment_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> Arc<TraitEnvironment> {
    let resolver = def.resolver(db.upcast());
    let mut ctx = if let GenericDefId::FunctionId(_) = def {
        TyLoweringContext::new(db, &resolver, TypesMap::EMPTY, def.into())
            .with_impl_trait_mode(ImplTraitLoweringMode::Param)
            .with_type_param_mode(ParamLoweringMode::Placeholder)
    } else {
        TyLoweringContext::new(db, &resolver, TypesMap::EMPTY, def.into())
            .with_type_param_mode(ParamLoweringMode::Placeholder)
    };
    let mut traits_in_scope = Vec::new();
    let mut clauses = Vec::new();
    for (params, def) in resolver.all_generic_params() {
        ctx.types_map = &params.types_map;
        for pred in params.where_predicates() {
            for pred in ctx.lower_where_predicate(pred, def, false) {
                if let WhereClause::Implemented(tr) = pred.skip_binders() {
                    traits_in_scope
                        .push((tr.self_type_parameter(Interner).clone(), tr.hir_trait_id()));
                }
                let program_clause: chalk_ir::ProgramClause<Interner> = pred.cast(Interner);
                clauses.push(program_clause.into_from_env_clause(Interner));
            }
        }
    }

    if let Some(trait_id) = def.assoc_trait_container(db.upcast()) {
        // add `Self: Trait<T1, T2, ...>` to the environment in trait
        // function default implementations (and speculative code
        // inside consts or type aliases)
        cov_mark::hit!(trait_self_implements_self);
        let substs = TyBuilder::placeholder_subst(db, trait_id);
        let trait_ref = TraitRef { trait_id: to_chalk_trait_id(trait_id), substitution: substs };
        let pred = WhereClause::Implemented(trait_ref);
        clauses.push(pred.cast::<ProgramClause>(Interner).into_from_env_clause(Interner));
    }

    let subst = generics(db.upcast(), def).placeholder_subst(db);
    if !subst.is_empty(Interner) {
        let explicitly_unsized_tys = ctx.unsized_types;
        if let Some(implicitly_sized_clauses) =
            implicitly_sized_clauses(db, def, &explicitly_unsized_tys, &subst, &resolver)
        {
            clauses.extend(
                implicitly_sized_clauses.map(|pred| {
                    pred.cast::<ProgramClause>(Interner).into_from_env_clause(Interner)
                }),
            );
        };
    }

    let env = chalk_ir::Environment::new(Interner).add_clauses(Interner, clauses);

    TraitEnvironment::new(resolver.krate(), None, traits_in_scope.into_boxed_slice(), env)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericPredicates(Option<Arc<[Binders<QuantifiedWhereClause>]>>);

impl ops::Deref for GenericPredicates {
    type Target = [Binders<crate::QuantifiedWhereClause>];

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().unwrap_or(&[])
    }
}

/// Resolve the where clause(s) of an item with generics.
pub(crate) fn generic_predicates_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> GenericPredicates {
    generic_predicates_filtered_by(db, def, |_, _| true).0
}

pub(crate) fn generic_predicates_without_parent_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> GenericPredicates {
    db.generic_predicates_without_parent_with_diagnostics(def).0
}

/// Resolve the where clause(s) of an item with generics,
/// except the ones inherited from the parent
pub(crate) fn generic_predicates_without_parent_with_diagnostics_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> (GenericPredicates, Diagnostics) {
    generic_predicates_filtered_by(db, def, |_, d| *d == def)
}

/// Resolve the where clause(s) of an item with generics,
/// except the ones inherited from the parent
fn generic_predicates_filtered_by<F>(
    db: &dyn HirDatabase,
    def: GenericDefId,
    filter: F,
) -> (GenericPredicates, Diagnostics)
where
    F: Fn(&WherePredicate, &GenericDefId) -> bool,
{
    let resolver = def.resolver(db.upcast());
    let (impl_trait_lowering, param_lowering) = match def {
        GenericDefId::FunctionId(_) => {
            (ImplTraitLoweringMode::Variable, ParamLoweringMode::Variable)
        }
        _ => (ImplTraitLoweringMode::Disallowed, ParamLoweringMode::Variable),
    };
    let mut ctx = TyLoweringContext::new(db, &resolver, TypesMap::EMPTY, def.into())
        .with_impl_trait_mode(impl_trait_lowering)
        .with_type_param_mode(param_lowering);
    let generics = generics(db.upcast(), def);

    let mut predicates = Vec::new();
    for (params, def) in resolver.all_generic_params() {
        ctx.types_map = &params.types_map;
        for pred in params.where_predicates() {
            if filter(pred, def) {
                predicates.extend(
                    ctx.lower_where_predicate(pred, def, false)
                        .map(|p| make_binders(db, &generics, p)),
                );
            }
        }
    }

    if generics.len() > 0 {
        let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
        let explicitly_unsized_tys = ctx.unsized_types;
        if let Some(implicitly_sized_predicates) =
            implicitly_sized_clauses(db, def, &explicitly_unsized_tys, &subst, &resolver)
        {
            predicates.extend(
                implicitly_sized_predicates
                    .map(|p| make_binders(db, &generics, crate::wrap_empty_binders(p))),
            );
        };
    }
    (
        GenericPredicates(predicates.is_empty().not().then(|| predicates.into())),
        create_diagnostics(ctx.diagnostics),
    )
}

/// Generate implicit `: Sized` predicates for all generics that has no `?Sized` bound.
/// Exception is Self of a trait def.
fn implicitly_sized_clauses<'a, 'subst: 'a>(
    db: &dyn HirDatabase,
    def: GenericDefId,
    explicitly_unsized_tys: &'a FxHashSet<Ty>,
    substitution: &'subst Substitution,
    resolver: &Resolver,
) -> Option<impl Iterator<Item = WhereClause> + Captures<'a> + Captures<'subst>> {
    let sized_trait = db
        .lang_item(resolver.krate(), LangItem::Sized)
        .and_then(|lang_item| lang_item.as_trait().map(to_chalk_trait_id))?;

    let trait_self_idx = trait_self_param_idx(db.upcast(), def);

    Some(
        substitution
            .iter(Interner)
            .enumerate()
            .filter_map(
                move |(idx, generic_arg)| {
                    if Some(idx) == trait_self_idx {
                        None
                    } else {
                        Some(generic_arg)
                    }
                },
            )
            .filter_map(|generic_arg| generic_arg.ty(Interner))
            .filter(move |&self_ty| !explicitly_unsized_tys.contains(self_ty))
            .map(move |self_ty| {
                WhereClause::Implemented(TraitRef {
                    trait_id: sized_trait,
                    substitution: Substitution::from1(Interner, self_ty.clone()),
                })
            }),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericDefaults(Option<Arc<[Binders<crate::GenericArg>]>>);

impl ops::Deref for GenericDefaults {
    type Target = [Binders<crate::GenericArg>];

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().unwrap_or(&[])
    }
}

pub(crate) fn generic_defaults_query(db: &dyn HirDatabase, def: GenericDefId) -> GenericDefaults {
    db.generic_defaults_with_diagnostics(def).0
}

/// Resolve the default type params from generics.
///
/// Diagnostics are only returned for this `GenericDefId` (returned defaults include parents).
pub(crate) fn generic_defaults_with_diagnostics_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> (GenericDefaults, Diagnostics) {
    let generic_params = generics(db.upcast(), def);
    if generic_params.len() == 0 {
        return (GenericDefaults(None), None);
    }
    let resolver = def.resolver(db.upcast());
    let parent_start_idx = generic_params.len_self();

    let mut ctx =
        TyLoweringContext::new(db, &resolver, generic_params.self_types_map(), def.into())
            .with_impl_trait_mode(ImplTraitLoweringMode::Disallowed)
            .with_type_param_mode(ParamLoweringMode::Variable);
    let mut idx = 0;
    let mut defaults = generic_params
        .iter_self()
        .map(|(id, p)| {
            let result =
                handle_generic_param(&mut ctx, idx, id, p, parent_start_idx, &generic_params);
            idx += 1;
            result
        })
        .collect::<Vec<_>>();
    let diagnostics = create_diagnostics(mem::take(&mut ctx.diagnostics));
    defaults.extend(generic_params.iter_parents_with_types_map().map(|((id, p), types_map)| {
        ctx.types_map = types_map;
        let result = handle_generic_param(&mut ctx, idx, id, p, parent_start_idx, &generic_params);
        idx += 1;
        result
    }));
    let defaults = GenericDefaults(Some(Arc::from_iter(defaults)));
    return (defaults, diagnostics);

    fn handle_generic_param(
        ctx: &mut TyLoweringContext<'_>,
        idx: usize,
        id: GenericParamId,
        p: GenericParamDataRef<'_>,
        parent_start_idx: usize,
        generic_params: &Generics,
    ) -> Binders<crate::GenericArg> {
        match p {
            GenericParamDataRef::TypeParamData(p) => {
                let ty = p.default.as_ref().map_or(TyKind::Error.intern(Interner), |ty| {
                    // Each default can only refer to previous parameters.
                    // Type variable default referring to parameter coming
                    // after it is forbidden (FIXME: report diagnostic)
                    fallback_bound_vars(ctx.lower_ty(*ty), idx, parent_start_idx)
                });
                crate::make_binders(ctx.db, generic_params, ty.cast(Interner))
            }
            GenericParamDataRef::ConstParamData(p) => {
                let GenericParamId::ConstParamId(id) = id else {
                    unreachable!("Unexpected lifetime or type argument")
                };

                let mut val = p.default.as_ref().map_or_else(
                    || unknown_const_as_generic(ctx.db.const_param_ty(id)),
                    |c| {
                        let param_ty = ctx.lower_ty(p.ty);
                        let c = ctx.lower_const(c, param_ty);
                        c.cast(Interner)
                    },
                );
                // Each default can only refer to previous parameters, see above.
                val = fallback_bound_vars(val, idx, parent_start_idx);
                make_binders(ctx.db, generic_params, val)
            }
            GenericParamDataRef::LifetimeParamData(_) => {
                make_binders(ctx.db, generic_params, error_lifetime().cast(Interner))
            }
        }
    }
}

pub(crate) fn generic_defaults_with_diagnostics_recover(
    db: &dyn HirDatabase,
    _cycle: &Cycle,
    def: &GenericDefId,
) -> (GenericDefaults, Diagnostics) {
    let generic_params = generics(db.upcast(), *def);
    if generic_params.len() == 0 {
        return (GenericDefaults(None), None);
    }
    // FIXME: this code is not covered in tests.
    // we still need one default per parameter
    let defaults = GenericDefaults(Some(Arc::from_iter(generic_params.iter_id().map(|id| {
        let val = match id {
            GenericParamId::TypeParamId(_) => TyKind::Error.intern(Interner).cast(Interner),
            GenericParamId::ConstParamId(id) => unknown_const_as_generic(db.const_param_ty(id)),
            GenericParamId::LifetimeParamId(_) => error_lifetime().cast(Interner),
        };
        crate::make_binders(db, &generic_params, val)
    }))));
    (defaults, None)
}

fn fn_sig_for_fn(db: &dyn HirDatabase, def: FunctionId) -> PolyFnSig {
    let data = db.function_data(def);
    let resolver = def.resolver(db.upcast());
    let mut ctx_params = TyLoweringContext::new(db, &resolver, &data.types_map, def.into())
        .with_impl_trait_mode(ImplTraitLoweringMode::Variable)
        .with_type_param_mode(ParamLoweringMode::Variable);
    let params = data.params.iter().map(|&tr| ctx_params.lower_ty(tr));
    let mut ctx_ret = TyLoweringContext::new(db, &resolver, &data.types_map, def.into())
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(ParamLoweringMode::Variable);
    let ret = ctx_ret.lower_ty(data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let sig = CallableSig::from_params_and_return(
        params,
        ret,
        data.is_varargs(),
        if data.is_unsafe() { Safety::Unsafe } else { Safety::Safe },
        data.abi.as_ref().map_or(FnAbi::Rust, FnAbi::from_symbol),
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
    let mut ctx = TyLoweringContext::new(db, &resolver, &data.types_map, def.into())
        .with_type_param_mode(ParamLoweringMode::Variable);

    make_binders(db, &generics, ctx.lower_ty(data.type_ref))
}

/// Build the declared type of a static.
fn type_for_static(db: &dyn HirDatabase, def: StaticId) -> Binders<Ty> {
    let data = db.static_data(def);
    let resolver = def.resolver(db.upcast());
    let mut ctx = TyLoweringContext::new(db, &resolver, &data.types_map, def.into());

    Binders::empty(Interner, ctx.lower_ty(data.type_ref))
}

fn fn_sig_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> PolyFnSig {
    let struct_data = db.struct_data(def);
    let fields = struct_data.variant_data.fields();
    let resolver = def.resolver(db.upcast());
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        struct_data.variant_data.types_map(),
        AdtId::from(def).into(),
    )
    .with_type_param_mode(ParamLoweringMode::Variable);
    let params = fields.iter().map(|(_, field)| ctx.lower_ty(field.type_ref));
    let (ret, binders) = type_for_adt(db, def.into()).into_value_and_skipped_binders();
    Binders::new(
        binders,
        CallableSig::from_params_and_return(params, ret, false, Safety::Safe, FnAbi::RustCall),
    )
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor(db: &dyn HirDatabase, def: StructId) -> Option<Binders<Ty>> {
    let struct_data = db.struct_data(def);
    match struct_data.variant_data.kind() {
        StructKind::Record => None,
        StructKind::Unit => Some(type_for_adt(db, def.into())),
        StructKind::Tuple => {
            let generics = generics(db.upcast(), AdtId::from(def).into());
            let substs = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
            Some(make_binders(
                db,
                &generics,
                TyKind::FnDef(CallableDefId::StructId(def).to_chalk(db), substs).intern(Interner),
            ))
        }
    }
}

fn fn_sig_for_enum_variant_constructor(db: &dyn HirDatabase, def: EnumVariantId) -> PolyFnSig {
    let var_data = db.enum_variant_data(def);
    let fields = var_data.variant_data.fields();
    let resolver = def.resolver(db.upcast());
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        var_data.variant_data.types_map(),
        DefWithBodyId::VariantId(def).into(),
    )
    .with_type_param_mode(ParamLoweringMode::Variable);
    let params = fields.iter().map(|(_, field)| ctx.lower_ty(field.type_ref));
    let (ret, binders) =
        type_for_adt(db, def.lookup(db.upcast()).parent.into()).into_value_and_skipped_binders();
    Binders::new(
        binders,
        CallableSig::from_params_and_return(params, ret, false, Safety::Safe, FnAbi::RustCall),
    )
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor(
    db: &dyn HirDatabase,
    def: EnumVariantId,
) -> Option<Binders<Ty>> {
    let e = def.lookup(db.upcast()).parent;
    match db.enum_variant_data(def).variant_data.kind() {
        StructKind::Record => None,
        StructKind::Unit => Some(type_for_adt(db, e.into())),
        StructKind::Tuple => {
            let generics = generics(db.upcast(), e.into());
            let substs = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
            Some(make_binders(
                db,
                &generics,
                TyKind::FnDef(CallableDefId::EnumVariantId(def).to_chalk(db), substs)
                    .intern(Interner),
            ))
        }
    }
}

fn type_for_adt(db: &dyn HirDatabase, adt: AdtId) -> Binders<Ty> {
    let generics = generics(db.upcast(), adt.into());
    let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    let ty = TyKind::Adt(crate::AdtId(adt), subst).intern(Interner);
    make_binders(db, &generics, ty)
}

pub(crate) fn type_for_type_alias_with_diagnostics_query(
    db: &dyn HirDatabase,
    t: TypeAliasId,
) -> (Binders<Ty>, Diagnostics) {
    let generics = generics(db.upcast(), t.into());
    let resolver = t.resolver(db.upcast());
    let type_alias_data = db.type_alias_data(t);
    let mut ctx = TyLoweringContext::new(db, &resolver, &type_alias_data.types_map, t.into())
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(ParamLoweringMode::Variable);
    let inner = if type_alias_data.is_extern {
        TyKind::Foreign(crate::to_foreign_def_id(t)).intern(Interner)
    } else {
        type_alias_data
            .type_ref
            .map(|type_ref| ctx.lower_ty(type_ref))
            .unwrap_or_else(|| TyKind::Error.intern(Interner))
    };
    (make_binders(db, &generics, inner), create_diagnostics(ctx.diagnostics))
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
    pub(crate) fn to_generic_def_id(self, db: &dyn HirDatabase) -> GenericDefId {
        match self {
            Self::FunctionId(id) => id.into(),
            Self::StructId(id) => id.into(),
            Self::UnionId(id) => id.into(),
            Self::EnumVariantId(var) => var.lookup(db.upcast()).parent.into(),
            Self::ConstId(id) => id.into(),
            Self::StaticId(id) => id.into(),
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
        TyDefId::TypeAliasId(it) => db.type_for_type_alias_with_diagnostics(it).0,
    }
}

pub(crate) fn ty_recover(db: &dyn HirDatabase, _cycle: &Cycle, def: &TyDefId) -> Binders<Ty> {
    let generics = match *def {
        TyDefId::BuiltinType(_) => return Binders::empty(Interner, TyKind::Error.intern(Interner)),
        TyDefId::AdtId(it) => generics(db.upcast(), it.into()),
        TyDefId::TypeAliasId(it) => generics(db.upcast(), it.into()),
    };
    make_binders(db, &generics, TyKind::Error.intern(Interner))
}

pub(crate) fn value_ty_query(db: &dyn HirDatabase, def: ValueTyDefId) -> Option<Binders<Ty>> {
    match def {
        ValueTyDefId::FunctionId(it) => Some(type_for_fn(db, it)),
        ValueTyDefId::StructId(it) => type_for_struct_constructor(db, it),
        ValueTyDefId::UnionId(it) => Some(type_for_adt(db, it.into())),
        ValueTyDefId::EnumVariantId(it) => type_for_enum_variant_constructor(db, it),
        ValueTyDefId::ConstId(it) => Some(type_for_const(db, it)),
        ValueTyDefId::StaticId(it) => Some(type_for_static(db, it)),
    }
}

pub(crate) fn impl_self_ty_query(db: &dyn HirDatabase, impl_id: ImplId) -> Binders<Ty> {
    db.impl_self_ty_with_diagnostics(impl_id).0
}

pub(crate) fn impl_self_ty_with_diagnostics_query(
    db: &dyn HirDatabase,
    impl_id: ImplId,
) -> (Binders<Ty>, Diagnostics) {
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db.upcast());
    let generics = generics(db.upcast(), impl_id.into());
    let mut ctx = TyLoweringContext::new(db, &resolver, &impl_data.types_map, impl_id.into())
        .with_type_param_mode(ParamLoweringMode::Variable);
    (
        make_binders(db, &generics, ctx.lower_ty(impl_data.self_ty)),
        create_diagnostics(ctx.diagnostics),
    )
}

pub(crate) fn const_param_ty_query(db: &dyn HirDatabase, def: ConstParamId) -> Ty {
    db.const_param_ty_with_diagnostics(def).0
}

// returns None if def is a type arg
pub(crate) fn const_param_ty_with_diagnostics_query(
    db: &dyn HirDatabase,
    def: ConstParamId,
) -> (Ty, Diagnostics) {
    let parent_data = db.generic_params(def.parent());
    let data = &parent_data[def.local_id()];
    let resolver = def.parent().resolver(db.upcast());
    let mut ctx =
        TyLoweringContext::new(db, &resolver, &parent_data.types_map, def.parent().into());
    let ty = match data {
        TypeOrConstParamData::TypeParamData(_) => {
            never!();
            Ty::new(Interner, TyKind::Error)
        }
        TypeOrConstParamData::ConstParamData(d) => ctx.lower_ty(d.ty),
    };
    (ty, create_diagnostics(ctx.diagnostics))
}

pub(crate) fn impl_self_ty_with_diagnostics_recover(
    db: &dyn HirDatabase,
    _cycle: &Cycle,
    impl_id: &ImplId,
) -> (Binders<Ty>, Diagnostics) {
    let generics = generics(db.upcast(), (*impl_id).into());
    (make_binders(db, &generics, TyKind::Error.intern(Interner)), None)
}

pub(crate) fn impl_trait_query(db: &dyn HirDatabase, impl_id: ImplId) -> Option<Binders<TraitRef>> {
    db.impl_trait_with_diagnostics(impl_id).map(|it| it.0)
}

pub(crate) fn impl_trait_with_diagnostics_query(
    db: &dyn HirDatabase,
    impl_id: ImplId,
) -> Option<(Binders<TraitRef>, Diagnostics)> {
    let impl_data = db.impl_data(impl_id);
    let resolver = impl_id.resolver(db.upcast());
    let mut ctx = TyLoweringContext::new(db, &resolver, &impl_data.types_map, impl_id.into())
        .with_type_param_mode(ParamLoweringMode::Variable);
    let (self_ty, binders) = db.impl_self_ty(impl_id).into_value_and_skipped_binders();
    let target_trait = impl_data.target_trait.as_ref()?;
    let trait_ref = Binders::new(binders, ctx.lower_trait_ref(target_trait, self_ty)?);
    Some((trait_ref, create_diagnostics(ctx.diagnostics)))
}

pub(crate) fn return_type_impl_traits(
    db: &dyn HirDatabase,
    def: hir_def::FunctionId,
) -> Option<Arc<Binders<ImplTraits>>> {
    // FIXME unify with fn_sig_for_fn instead of doing lowering twice, maybe
    let data = db.function_data(def);
    let resolver = def.resolver(db.upcast());
    let mut ctx_ret = TyLoweringContext::new(db, &resolver, &data.types_map, def.into())
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(ParamLoweringMode::Variable);
    let _ret = ctx_ret.lower_ty(data.ret_type);
    let generics = generics(db.upcast(), def.into());
    let return_type_impl_traits =
        ImplTraits { impl_traits: ctx_ret.impl_trait_mode.opaque_type_data };
    if return_type_impl_traits.impl_traits.is_empty() {
        None
    } else {
        Some(Arc::new(make_binders(db, &generics, return_type_impl_traits)))
    }
}

pub(crate) fn type_alias_impl_traits(
    db: &dyn HirDatabase,
    def: hir_def::TypeAliasId,
) -> Option<Arc<Binders<ImplTraits>>> {
    let data = db.type_alias_data(def);
    let resolver = def.resolver(db.upcast());
    let mut ctx = TyLoweringContext::new(db, &resolver, &data.types_map, def.into())
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
        .with_type_param_mode(ParamLoweringMode::Variable);
    if let Some(type_ref) = data.type_ref {
        let _ty = ctx.lower_ty(type_ref);
    }
    let type_alias_impl_traits = ImplTraits { impl_traits: ctx.impl_trait_mode.opaque_type_data };
    if type_alias_impl_traits.impl_traits.is_empty() {
        None
    } else {
        let generics = generics(db.upcast(), def.into());
        Some(Arc::new(make_binders(db, &generics, type_alias_impl_traits)))
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
pub(crate) fn generic_arg_to_chalk<'a, T>(
    db: &dyn HirDatabase,
    kind_id: GenericParamId,
    arg: &'a GenericArg,
    this: &mut T,
    types_map: &TypesMap,
    for_type: impl FnOnce(&mut T, TypeRefId) -> Ty + 'a,
    for_const: impl FnOnce(&mut T, &ConstRef, Ty) -> Const + 'a,
    for_lifetime: impl FnOnce(&mut T, &LifetimeRef) -> Lifetime + 'a,
) -> crate::GenericArg {
    let kind = match kind_id {
        GenericParamId::TypeParamId(_) => ParamKind::Type,
        GenericParamId::ConstParamId(id) => {
            let ty = db.const_param_ty(id);
            ParamKind::Const(ty)
        }
        GenericParamId::LifetimeParamId(_) => ParamKind::Lifetime,
    };
    match (arg, kind) {
        (GenericArg::Type(type_ref), ParamKind::Type) => for_type(this, *type_ref).cast(Interner),
        (GenericArg::Const(c), ParamKind::Const(c_ty)) => for_const(this, c, c_ty).cast(Interner),
        (GenericArg::Lifetime(lifetime_ref), ParamKind::Lifetime) => {
            for_lifetime(this, lifetime_ref).cast(Interner)
        }
        (GenericArg::Const(_), ParamKind::Type) => TyKind::Error.intern(Interner).cast(Interner),
        (GenericArg::Lifetime(_), ParamKind::Type) => TyKind::Error.intern(Interner).cast(Interner),
        (GenericArg::Type(t), ParamKind::Const(c_ty)) => {
            // We want to recover simple idents, which parser detects them
            // as types. Maybe here is not the best place to do it, but
            // it works.
            if let TypeRef::Path(p) = &types_map[*t] {
                if let Some(p) = p.mod_path() {
                    if p.kind == PathKind::Plain {
                        if let [n] = p.segments() {
                            let c = ConstRef::Path(n.clone());
                            return for_const(this, &c, c_ty).cast(Interner);
                        }
                    }
                }
            }
            unknown_const_as_generic(c_ty)
        }
        (GenericArg::Lifetime(_), ParamKind::Const(c_ty)) => unknown_const_as_generic(c_ty),
        (GenericArg::Type(_), ParamKind::Lifetime) => error_lifetime().cast(Interner),
        (GenericArg::Const(_), ParamKind::Lifetime) => error_lifetime().cast(Interner),
    }
}

pub(crate) fn const_or_path_to_chalk<'g>(
    db: &dyn HirDatabase,
    resolver: &Resolver,
    owner: TypeOwnerId,
    expected_ty: Ty,
    value: &ConstRef,
    mode: ParamLoweringMode,
    args: impl FnOnce() -> Option<&'g Generics>,
    debruijn: DebruijnIndex,
) -> Const {
    match value {
        ConstRef::Scalar(s) => intern_const_ref(db, s, expected_ty, resolver.krate()),
        ConstRef::Path(n) => {
            let path = ModPath::from_segments(PathKind::Plain, Some(n.clone()));
            path_to_const(
                db,
                resolver,
                &Path::from_known_path_with_no_generic(path),
                mode,
                args,
                debruijn,
                expected_ty.clone(),
            )
            .unwrap_or_else(|| unknown_const(expected_ty))
        }
        &ConstRef::Complex(it) => {
            let crate_data = &db.crate_graph()[resolver.krate()];
            if crate_data.env.get("__ra_is_test_fixture").is_none() && crate_data.origin.is_local()
            {
                // FIXME: current `InTypeConstId` is very unstable, so we only use it in non local crate
                // that are unlikely to be edited.
                return unknown_const(expected_ty);
            }
            let c = db
                .intern_in_type_const(InTypeConstLoc {
                    id: it,
                    owner,
                    expected_ty: Box::new(InTypeConstIdMetadata(expected_ty.clone())),
                })
                .into();
            intern_const_scalar(
                ConstScalar::UnevaluatedConst(c, Substitution::empty(Interner)),
                expected_ty,
            )
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
            if bound.index_if_innermost().is_none_or(is_allowed) {
                bound.shifted_in_from(binders).to_ty(Interner)
            } else {
                TyKind::Error.intern(Interner)
            }
        },
        |ty, bound, binders| {
            if bound.index_if_innermost().is_none_or(is_allowed) {
                bound.shifted_in_from(binders).to_const(Interner, ty)
            } else {
                unknown_const(ty)
            }
        },
    )
}
