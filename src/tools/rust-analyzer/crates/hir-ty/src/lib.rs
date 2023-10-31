//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.
#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod builder;
mod chalk_db;
mod chalk_ext;
mod infer;
mod inhabitedness;
mod interner;
mod lower;
mod mapping;
mod tls;
mod utils;

pub mod autoderef;
pub mod consteval;
pub mod db;
pub mod diagnostics;
pub mod display;
pub mod lang_items;
pub mod layout;
pub mod method_resolution;
pub mod mir;
pub mod primitive;
pub mod traits;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod test_db;

use std::{
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
};

use chalk_ir::{
    fold::{Shift, TypeFoldable},
    interner::HasInterner,
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
    NoSolution, TyData,
};
use either::Either;
use hir_def::{hir::ExprId, type_ref::Rawness, GeneralConstId, TypeOrConstParamId};
use hir_expand::name;
use la_arena::{Arena, Idx};
use mir::{MirEvalError, VTableMap};
use rustc_hash::FxHashSet;
use traits::FnTrait;
use triomphe::Arc;
use utils::Generics;

use crate::{
    consteval::unknown_const, db::HirDatabase, infer::unify::InferenceTable, utils::generics,
};

pub use autoderef::autoderef;
pub use builder::{ParamKind, TyBuilder};
pub use chalk_ext::*;
pub use infer::{
    closure::{CaptureKind, CapturedItem},
    could_coerce, could_unify, Adjust, Adjustment, AutoBorrow, BindingMode, InferenceDiagnostic,
    InferenceResult, OverloadedDeref, PointerCast,
};
pub use interner::Interner;
pub use lower::{
    associated_type_shorthand_candidates, CallableDefId, ImplTraitLoweringMode, TyDefId,
    TyLoweringContext, ValueTyDefId,
};
pub use mapping::{
    from_assoc_type_id, from_chalk_trait_id, from_foreign_def_id, from_placeholder_idx,
    lt_from_placeholder_idx, to_assoc_type_id, to_chalk_trait_id, to_foreign_def_id,
    to_placeholder_idx,
};
pub use traits::TraitEnvironment;
pub use utils::{all_super_traits, is_fn_unsafe_to_call};

pub use chalk_ir::{
    cast::Cast, AdtId, BoundVar, DebruijnIndex, Mutability, Safety, Scalar, TyVariableKind,
};

pub type ForeignDefId = chalk_ir::ForeignDefId<Interner>;
pub type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub type FnDefId = chalk_ir::FnDefId<Interner>;
pub type ClosureId = chalk_ir::ClosureId<Interner>;
pub type OpaqueTyId = chalk_ir::OpaqueTyId<Interner>;
pub type PlaceholderIndex = chalk_ir::PlaceholderIndex;

pub type VariableKind = chalk_ir::VariableKind<Interner>;
pub type VariableKinds = chalk_ir::VariableKinds<Interner>;
pub type CanonicalVarKinds = chalk_ir::CanonicalVarKinds<Interner>;
/// Represents generic parameters and an item bound by them. When the item has parent, the binders
/// also contain the generic parameters for its parent. See chalk's documentation for details.
///
/// One thing to keep in mind when working with `Binders` (and `Substitution`s, which represent
/// generic arguments) in rust-analyzer is that the ordering within *is* significant - the generic
/// parameters/arguments for an item MUST come before those for its parent. This is to facilitate
/// the integration with chalk-solve, which mildly puts constraints as such. See #13335 for its
/// motivation in detail.
pub type Binders<T> = chalk_ir::Binders<T>;
/// Interned list of generic arguments for an item. When an item has parent, the `Substitution` for
/// it contains generic arguments for both its parent and itself. See chalk's documentation for
/// details.
///
/// See `Binders` for the constraint on the ordering.
pub type Substitution = chalk_ir::Substitution<Interner>;
pub type GenericArg = chalk_ir::GenericArg<Interner>;
pub type GenericArgData = chalk_ir::GenericArgData<Interner>;

pub type Ty = chalk_ir::Ty<Interner>;
pub type TyKind = chalk_ir::TyKind<Interner>;
pub type TypeFlags = chalk_ir::TypeFlags;
pub type DynTy = chalk_ir::DynTy<Interner>;
pub type FnPointer = chalk_ir::FnPointer<Interner>;
// pub type FnSubst = chalk_ir::FnSubst<Interner>;
pub use chalk_ir::FnSubst;
pub type ProjectionTy = chalk_ir::ProjectionTy<Interner>;
pub type AliasTy = chalk_ir::AliasTy<Interner>;
pub type OpaqueTy = chalk_ir::OpaqueTy<Interner>;
pub type InferenceVar = chalk_ir::InferenceVar;

pub type Lifetime = chalk_ir::Lifetime<Interner>;
pub type LifetimeData = chalk_ir::LifetimeData<Interner>;
pub type LifetimeOutlives = chalk_ir::LifetimeOutlives<Interner>;

pub type Const = chalk_ir::Const<Interner>;
pub type ConstData = chalk_ir::ConstData<Interner>;
pub type ConstValue = chalk_ir::ConstValue<Interner>;
pub type ConcreteConst = chalk_ir::ConcreteConst<Interner>;

pub type ChalkTraitId = chalk_ir::TraitId<Interner>;
pub type TraitRef = chalk_ir::TraitRef<Interner>;
pub type QuantifiedWhereClause = Binders<WhereClause>;
pub type QuantifiedWhereClauses = chalk_ir::QuantifiedWhereClauses<Interner>;
pub type Canonical<T> = chalk_ir::Canonical<T>;

pub type FnSig = chalk_ir::FnSig<Interner>;

pub type InEnvironment<T> = chalk_ir::InEnvironment<T>;
pub type Environment = chalk_ir::Environment<Interner>;
pub type DomainGoal = chalk_ir::DomainGoal<Interner>;
pub type Goal = chalk_ir::Goal<Interner>;
pub type AliasEq = chalk_ir::AliasEq<Interner>;
pub type Solution = chalk_solve::Solution<Interner>;
pub type ConstrainedSubst = chalk_ir::ConstrainedSubst<Interner>;
pub type Guidance = chalk_solve::Guidance<Interner>;
pub type WhereClause = chalk_ir::WhereClause<Interner>;

/// A constant can have reference to other things. Memory map job is holding
/// the necessary bits of memory of the const eval session to keep the constant
/// meaningful.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MemoryMap {
    pub memory: HashMap<usize, Vec<u8>>,
    pub vtable: VTableMap,
}

impl MemoryMap {
    fn insert(&mut self, addr: usize, x: Vec<u8>) {
        match self.memory.entry(addr) {
            Entry::Occupied(mut e) => {
                if e.get().len() < x.len() {
                    e.insert(x);
                }
            }
            Entry::Vacant(e) => {
                e.insert(x);
            }
        }
    }

    /// This functions convert each address by a function `f` which gets the byte intervals and assign an address
    /// to them. It is useful when you want to load a constant with a memory map in a new memory. You can pass an
    /// allocator function as `f` and it will return a mapping of old addresses to new addresses.
    fn transform_addresses(
        &self,
        mut f: impl FnMut(&[u8], usize) -> Result<usize, MirEvalError>,
    ) -> Result<HashMap<usize, usize>, MirEvalError> {
        self.memory
            .iter()
            .map(|x| {
                let addr = *x.0;
                let align = if addr == 0 { 64 } else { (addr - (addr & (addr - 1))).min(64) };
                Ok((addr, f(x.1, align)?))
            })
            .collect()
    }

    fn get<'a>(&'a self, addr: usize, size: usize) -> Option<&'a [u8]> {
        if size == 0 {
            Some(&[])
        } else {
            self.memory.get(&addr)?.get(0..size)
        }
    }
}

/// A concrete constant value
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstScalar {
    Bytes(Vec<u8>, MemoryMap),
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    UnevaluatedConst(GeneralConstId, Substitution),
    /// Case of an unknown value that rustc might know but we don't
    // FIXME: this is a hack to get around chalk not being able to represent unevaluatable
    // constants
    // https://github.com/rust-lang/rust-analyzer/pull/8813#issuecomment-840679177
    // https://rust-lang.zulipchat.com/#narrow/stream/144729-wg-traits/topic/Handling.20non.20evaluatable.20constants'.20equality/near/238386348
    Unknown,
}

impl Hash for ConstScalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        if let ConstScalar::Bytes(b, _) = self {
            b.hash(state)
        }
    }
}

/// Return an index of a parameter in the generic type parameter list by it's id.
pub fn param_idx(db: &dyn HirDatabase, id: TypeOrConstParamId) -> Option<usize> {
    generics(db.upcast(), id.parent).param_idx(id)
}

pub(crate) fn wrap_empty_binders<T>(value: T) -> Binders<T>
where
    T: TypeFoldable<Interner> + HasInterner<Interner = Interner>,
{
    Binders::empty(Interner, value.shifted_in_from(Interner, DebruijnIndex::ONE))
}

pub(crate) fn make_type_and_const_binders<T: HasInterner<Interner = Interner>>(
    which_is_const: impl Iterator<Item = Option<Ty>>,
    value: T,
) -> Binders<T> {
    Binders::new(
        VariableKinds::from_iter(
            Interner,
            which_is_const.map(|x| {
                if let Some(ty) = x {
                    chalk_ir::VariableKind::Const(ty)
                } else {
                    chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General)
                }
            }),
        ),
        value,
    )
}

pub(crate) fn make_single_type_binders<T: HasInterner<Interner = Interner>>(
    value: T,
) -> Binders<T> {
    Binders::new(
        VariableKinds::from_iter(
            Interner,
            std::iter::once(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General)),
        ),
        value,
    )
}

pub(crate) fn make_binders_with_count<T: HasInterner<Interner = Interner>>(
    db: &dyn HirDatabase,
    count: usize,
    generics: &Generics,
    value: T,
) -> Binders<T> {
    let it = generics.iter_id().take(count).map(|id| match id {
        Either::Left(_) => None,
        Either::Right(id) => Some(db.const_param_ty(id)),
    });
    crate::make_type_and_const_binders(it, value)
}

pub(crate) fn make_binders<T: HasInterner<Interner = Interner>>(
    db: &dyn HirDatabase,
    generics: &Generics,
    value: T,
) -> Binders<T> {
    make_binders_with_count(db, usize::MAX, generics, value)
}

// FIXME: get rid of this, just replace it by FnPointer
/// A function signature as seen by type inference: Several parameter types and
/// one return type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CallableSig {
    params_and_return: Arc<[Ty]>,
    is_varargs: bool,
    safety: Safety,
}

has_interner!(CallableSig);

/// A polymorphic function signature.
pub type PolyFnSig = Binders<CallableSig>;

impl CallableSig {
    pub fn from_params_and_return(
        mut params: Vec<Ty>,
        ret: Ty,
        is_varargs: bool,
        safety: Safety,
    ) -> CallableSig {
        params.push(ret);
        CallableSig { params_and_return: params.into(), is_varargs, safety }
    }

    pub fn from_fn_ptr(fn_ptr: &FnPointer) -> CallableSig {
        CallableSig {
            // FIXME: what to do about lifetime params? -> return PolyFnSig
            // FIXME: use `Arc::from_iter` when it becomes available
            params_and_return: Arc::from(
                fn_ptr
                    .substitution
                    .clone()
                    .shifted_out_to(Interner, DebruijnIndex::ONE)
                    .expect("unexpected lifetime vars in fn ptr")
                    .0
                    .as_slice(Interner)
                    .iter()
                    .map(|arg| arg.assert_ty_ref(Interner).clone())
                    .collect::<Vec<_>>(),
            ),
            is_varargs: fn_ptr.sig.variadic,
            safety: fn_ptr.sig.safety,
        }
    }

    pub fn to_fn_ptr(&self) -> FnPointer {
        FnPointer {
            num_binders: 0,
            sig: FnSig { abi: (), safety: self.safety, variadic: self.is_varargs },
            substitution: FnSubst(Substitution::from_iter(
                Interner,
                self.params_and_return.iter().cloned(),
            )),
        }
    }

    pub fn params(&self) -> &[Ty] {
        &self.params_and_return[0..self.params_and_return.len() - 1]
    }

    pub fn ret(&self) -> &Ty {
        &self.params_and_return[self.params_and_return.len() - 1]
    }
}

impl TypeFoldable<Interner> for CallableSig {
    fn try_fold_with<E>(
        self,
        folder: &mut dyn chalk_ir::fold::FallibleTypeFolder<Interner, Error = E>,
        outer_binder: DebruijnIndex,
    ) -> Result<Self, E> {
        let vec = self.params_and_return.to_vec();
        let folded = vec.try_fold_with(folder, outer_binder)?;
        Ok(CallableSig {
            params_and_return: folded.into(),
            is_varargs: self.is_varargs,
            safety: self.safety,
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ImplTraitId {
    ReturnTypeImplTrait(hir_def::FunctionId, RpitId),
    AsyncBlockTypeImplTrait(hir_def::DefWithBodyId, ExprId),
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ReturnTypeImplTraits {
    pub(crate) impl_traits: Arena<ReturnTypeImplTrait>,
}

has_interner!(ReturnTypeImplTraits);

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ReturnTypeImplTrait {
    pub(crate) bounds: Binders<Vec<QuantifiedWhereClause>>,
}

pub type RpitId = Idx<ReturnTypeImplTrait>;

pub fn static_lifetime() -> Lifetime {
    LifetimeData::Static.intern(Interner)
}

pub(crate) fn fold_free_vars<T: HasInterner<Interner = Interner> + TypeFoldable<Interner>>(
    t: T,
    for_ty: impl FnMut(BoundVar, DebruijnIndex) -> Ty,
    for_const: impl FnMut(Ty, BoundVar, DebruijnIndex) -> Const,
) -> T {
    use chalk_ir::fold::TypeFolder;

    #[derive(chalk_derive::FallibleTypeFolder)]
    #[has_interner(Interner)]
    struct FreeVarFolder<
        F1: FnMut(BoundVar, DebruijnIndex) -> Ty,
        F2: FnMut(Ty, BoundVar, DebruijnIndex) -> Const,
    >(F1, F2);
    impl<
            F1: FnMut(BoundVar, DebruijnIndex) -> Ty,
            F2: FnMut(Ty, BoundVar, DebruijnIndex) -> Const,
        > TypeFolder<Interner> for FreeVarFolder<F1, F2>
    {
        fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner, Error = Self::Error> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn fold_free_var_ty(&mut self, bound_var: BoundVar, outer_binder: DebruijnIndex) -> Ty {
            self.0(bound_var, outer_binder)
        }

        fn fold_free_var_const(
            &mut self,
            ty: Ty,
            bound_var: BoundVar,
            outer_binder: DebruijnIndex,
        ) -> Const {
            self.1(ty, bound_var, outer_binder)
        }
    }
    t.fold_with(&mut FreeVarFolder(for_ty, for_const), DebruijnIndex::INNERMOST)
}

pub(crate) fn fold_tys<T: HasInterner<Interner = Interner> + TypeFoldable<Interner>>(
    t: T,
    mut for_ty: impl FnMut(Ty, DebruijnIndex) -> Ty,
    binders: DebruijnIndex,
) -> T {
    fold_tys_and_consts(
        t,
        |x, d| match x {
            Either::Left(x) => Either::Left(for_ty(x, d)),
            Either::Right(x) => Either::Right(x),
        },
        binders,
    )
}

pub(crate) fn fold_tys_and_consts<T: HasInterner<Interner = Interner> + TypeFoldable<Interner>>(
    t: T,
    f: impl FnMut(Either<Ty, Const>, DebruijnIndex) -> Either<Ty, Const>,
    binders: DebruijnIndex,
) -> T {
    use chalk_ir::fold::{TypeFolder, TypeSuperFoldable};
    #[derive(chalk_derive::FallibleTypeFolder)]
    #[has_interner(Interner)]
    struct TyFolder<F: FnMut(Either<Ty, Const>, DebruijnIndex) -> Either<Ty, Const>>(F);
    impl<F: FnMut(Either<Ty, Const>, DebruijnIndex) -> Either<Ty, Const>> TypeFolder<Interner>
        for TyFolder<F>
    {
        fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner, Error = Self::Error> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn fold_ty(&mut self, ty: Ty, outer_binder: DebruijnIndex) -> Ty {
            let ty = ty.super_fold_with(self.as_dyn(), outer_binder);
            self.0(Either::Left(ty), outer_binder).left().unwrap()
        }

        fn fold_const(&mut self, c: Const, outer_binder: DebruijnIndex) -> Const {
            self.0(Either::Right(c), outer_binder).right().unwrap()
        }
    }
    t.fold_with(&mut TyFolder(f), binders)
}

/// 'Canonicalizes' the `t` by replacing any errors with new variables. Also
/// ensures there are no unbound variables or inference variables anywhere in
/// the `t`.
pub fn replace_errors_with_variables<T>(t: &T) -> Canonical<T>
where
    T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + Clone,
{
    use chalk_ir::{
        fold::{FallibleTypeFolder, TypeSuperFoldable},
        Fallible,
    };
    struct ErrorReplacer {
        vars: usize,
    }
    impl FallibleTypeFolder<Interner> for ErrorReplacer {
        type Error = NoSolution;

        fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = Self::Error> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn try_fold_ty(&mut self, ty: Ty, outer_binder: DebruijnIndex) -> Fallible<Ty> {
            if let TyKind::Error = ty.kind(Interner) {
                let index = self.vars;
                self.vars += 1;
                Ok(TyKind::BoundVar(BoundVar::new(outer_binder, index)).intern(Interner))
            } else {
                ty.try_super_fold_with(self.as_dyn(), outer_binder)
            }
        }

        fn try_fold_inference_ty(
            &mut self,
            _var: InferenceVar,
            _kind: TyVariableKind,
            _outer_binder: DebruijnIndex,
        ) -> Fallible<Ty> {
            if cfg!(debug_assertions) {
                // we don't want to just panic here, because then the error message
                // won't contain the whole thing, which would not be very helpful
                Err(NoSolution)
            } else {
                Ok(TyKind::Error.intern(Interner))
            }
        }

        fn try_fold_free_var_ty(
            &mut self,
            _bound_var: BoundVar,
            _outer_binder: DebruijnIndex,
        ) -> Fallible<Ty> {
            if cfg!(debug_assertions) {
                // we don't want to just panic here, because then the error message
                // won't contain the whole thing, which would not be very helpful
                Err(NoSolution)
            } else {
                Ok(TyKind::Error.intern(Interner))
            }
        }

        fn try_fold_inference_const(
            &mut self,
            ty: Ty,
            _var: InferenceVar,
            _outer_binder: DebruijnIndex,
        ) -> Fallible<Const> {
            if cfg!(debug_assertions) {
                Err(NoSolution)
            } else {
                Ok(unknown_const(ty))
            }
        }

        fn try_fold_free_var_const(
            &mut self,
            ty: Ty,
            _bound_var: BoundVar,
            _outer_binder: DebruijnIndex,
        ) -> Fallible<Const> {
            if cfg!(debug_assertions) {
                Err(NoSolution)
            } else {
                Ok(unknown_const(ty))
            }
        }

        fn try_fold_inference_lifetime(
            &mut self,
            _var: InferenceVar,
            _outer_binder: DebruijnIndex,
        ) -> Fallible<Lifetime> {
            if cfg!(debug_assertions) {
                Err(NoSolution)
            } else {
                Ok(static_lifetime())
            }
        }

        fn try_fold_free_var_lifetime(
            &mut self,
            _bound_var: BoundVar,
            _outer_binder: DebruijnIndex,
        ) -> Fallible<Lifetime> {
            if cfg!(debug_assertions) {
                Err(NoSolution)
            } else {
                Ok(static_lifetime())
            }
        }
    }
    let mut error_replacer = ErrorReplacer { vars: 0 };
    let value = match t.clone().try_fold_with(&mut error_replacer, DebruijnIndex::INNERMOST) {
        Ok(t) => t,
        Err(_) => panic!("Encountered unbound or inference vars in {t:?}"),
    };
    let kinds = (0..error_replacer.vars).map(|_| {
        chalk_ir::CanonicalVarKind::new(
            chalk_ir::VariableKind::Ty(TyVariableKind::General),
            chalk_ir::UniverseIndex::ROOT,
        )
    });
    Canonical { value, binders: chalk_ir::CanonicalVarKinds::from_iter(Interner, kinds) }
}

pub fn callable_sig_from_fnonce(
    mut self_ty: &Ty,
    env: Arc<TraitEnvironment>,
    db: &dyn HirDatabase,
) -> Option<CallableSig> {
    if let Some((ty, _, _)) = self_ty.as_reference() {
        // This will happen when it implements fn or fn mut, since we add a autoborrow adjustment
        self_ty = ty;
    }
    let krate = env.krate;
    let fn_once_trait = FnTrait::FnOnce.get_id(db, krate)?;
    let output_assoc_type = db.trait_data(fn_once_trait).associated_type_by_name(&name![Output])?;

    let mut table = InferenceTable::new(db, env);
    let b = TyBuilder::trait_ref(db, fn_once_trait);
    if b.remaining() != 2 {
        return None;
    }

    // Register two obligations:
    // - Self: FnOnce<?args_ty>
    // - <Self as FnOnce<?args_ty>>::Output == ?ret_ty
    let args_ty = table.new_type_var();
    let trait_ref = b.push(self_ty.clone()).push(args_ty.clone()).build();
    let projection = TyBuilder::assoc_type_projection(
        db,
        output_assoc_type,
        Some(trait_ref.substitution.clone()),
    )
    .build();
    table.register_obligation(trait_ref.cast(Interner));
    let ret_ty = table.normalize_projection_ty(projection);

    let ret_ty = table.resolve_completely(ret_ty);
    let args_ty = table.resolve_completely(args_ty);

    let params =
        args_ty.as_tuple()?.iter(Interner).map(|it| it.assert_ty_ref(Interner)).cloned().collect();

    Some(CallableSig::from_params_and_return(params, ret_ty, false, Safety::Safe))
}

struct PlaceholderCollector<'db> {
    db: &'db dyn HirDatabase,
    placeholders: FxHashSet<TypeOrConstParamId>,
}

impl PlaceholderCollector<'_> {
    fn collect(&mut self, idx: PlaceholderIndex) {
        let id = from_placeholder_idx(self.db, idx);
        self.placeholders.insert(id);
    }
}

impl TypeVisitor<Interner> for PlaceholderCollector<'_> {
    type BreakTy = ();

    fn as_dyn(&mut self) -> &mut dyn TypeVisitor<Interner, BreakTy = Self::BreakTy> {
        self
    }

    fn interner(&self) -> Interner {
        Interner
    }

    fn visit_ty(
        &mut self,
        ty: &Ty,
        outer_binder: DebruijnIndex,
    ) -> std::ops::ControlFlow<Self::BreakTy> {
        let has_placeholder_bits = TypeFlags::HAS_TY_PLACEHOLDER | TypeFlags::HAS_CT_PLACEHOLDER;
        let TyData { kind, flags } = ty.data(Interner);

        if let TyKind::Placeholder(idx) = kind {
            self.collect(*idx);
        } else if flags.intersects(has_placeholder_bits) {
            return ty.super_visit_with(self, outer_binder);
        } else {
            // Fast path: don't visit inner types (e.g. generic arguments) when `flags` indicate
            // that there are no placeholders.
        }

        std::ops::ControlFlow::Continue(())
    }

    fn visit_const(
        &mut self,
        constant: &chalk_ir::Const<Interner>,
        _outer_binder: DebruijnIndex,
    ) -> std::ops::ControlFlow<Self::BreakTy> {
        if let chalk_ir::ConstValue::Placeholder(idx) = constant.data(Interner).value {
            self.collect(idx);
        }
        std::ops::ControlFlow::Continue(())
    }
}

/// Returns unique placeholders for types and consts contained in `value`.
pub fn collect_placeholders<T>(value: &T, db: &dyn HirDatabase) -> Vec<TypeOrConstParamId>
where
    T: ?Sized + TypeVisitable<Interner>,
{
    let mut collector = PlaceholderCollector { db, placeholders: FxHashSet::default() };
    value.visit_with(&mut collector, DebruijnIndex::INNERMOST);
    collector.placeholders.into_iter().collect()
}
