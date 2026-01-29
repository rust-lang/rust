//! Things related to tys in the next-trait-solver.

use std::ops::ControlFlow;

use hir_def::{
    AdtId, HasModule, TypeParamId,
    hir::generics::{TypeOrConstParamData, TypeParamProvenance},
};
use hir_def::{TraitId, type_ref::Rawness};
use intern::{Interned, InternedRef, impl_internable};
use macros::GenericTypeVisitable;
use rustc_abi::{Float, Integer, Size};
use rustc_ast_ir::{Mutability, try_visit, visit::VisitorResult};
use rustc_type_ir::{
    AliasTyKind, BoundVar, BoundVarIndexKind, ClosureKind, CoroutineArgs, CoroutineArgsParts,
    DebruijnIndex, FlagComputation, Flags, FloatTy, FloatVid, GenericTypeVisitable, InferTy, IntTy,
    IntVid, Interner, TyVid, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor, UintTy, Upcast, WithCachedTypeInfo,
    inherent::{
        AdtDef as _, BoundExistentialPredicates, BoundVarLike, Const as _, GenericArgs as _,
        IntoKind, ParamLike, PlaceholderLike, Safety as _, SliceLike, Ty as _,
    },
    relate::Relate,
    solve::SizedTraitKind,
    walk::TypeWalker,
};

use crate::{
    FnAbi,
    db::{HirDatabase, InternedCoroutine},
    lower::GenericPredicates,
    next_solver::{
        AdtDef, AliasTy, Binder, CallableIdWrapper, Clause, ClauseKind, ClosureIdWrapper, Const,
        CoroutineIdWrapper, FnSig, GenericArgKind, PolyFnSig, Region, TraitRef, TypeAliasIdWrapper,
        abi::Safety,
        impl_foldable_for_interned_slice, impl_stored_interned, interned_slice,
        util::{CoroutineArgsExt, IntegerTypeExt},
    },
};

use super::{
    BoundVarKind, DbInterner, GenericArgs, Placeholder, SolverDefId,
    util::{FloatExt, IntegerExt},
};

pub type SimplifiedType = rustc_type_ir::fast_reject::SimplifiedType<SolverDefId>;
pub type TyKind<'db> = rustc_type_ir::TyKind<DbInterner<'db>>;
pub type FnHeader<'db> = rustc_type_ir::FnHeader<DbInterner<'db>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ty<'db> {
    pub(super) interned: InternedRef<'db, TyInterned>,
}

#[derive(PartialEq, Eq, Hash, GenericTypeVisitable)]
#[repr(align(4))] // Required for `GenericArg` bit-tagging.
pub(super) struct TyInterned(WithCachedTypeInfo<TyKind<'static>>);

impl_internable!(gc; TyInterned);
impl_stored_interned!(TyInterned, Ty, StoredTy);

const _: () = {
    const fn is_copy<T: Copy>() {}
    is_copy::<Ty<'static>>();
};

impl<'db> Ty<'db> {
    #[inline]
    pub fn new(_interner: DbInterner<'db>, kind: TyKind<'db>) -> Self {
        let kind = unsafe { std::mem::transmute::<TyKind<'db>, TyKind<'static>>(kind) };
        let flags = FlagComputation::for_kind(&kind);
        let cached = WithCachedTypeInfo {
            internee: kind,
            flags: flags.flags,
            outer_exclusive_binder: flags.outer_exclusive_binder,
        };
        Self { interned: Interned::new_gc(TyInterned(cached)) }
    }

    #[inline]
    pub fn inner(&self) -> &WithCachedTypeInfo<TyKind<'db>> {
        let inner = &self.interned.0;
        unsafe {
            std::mem::transmute::<
                &WithCachedTypeInfo<TyKind<'static>>,
                &WithCachedTypeInfo<TyKind<'db>>,
            >(inner)
        }
    }

    pub fn new_adt(interner: DbInterner<'db>, adt_id: AdtId, args: GenericArgs<'db>) -> Self {
        Ty::new(interner, TyKind::Adt(AdtDef::new(adt_id, interner), args))
    }

    pub fn new_param(interner: DbInterner<'db>, id: TypeParamId, index: u32) -> Self {
        Ty::new(interner, TyKind::Param(ParamTy { id, index }))
    }

    pub fn new_placeholder(interner: DbInterner<'db>, placeholder: PlaceholderTy) -> Self {
        Ty::new(interner, TyKind::Placeholder(placeholder))
    }

    pub fn new_infer(interner: DbInterner<'db>, infer: InferTy) -> Self {
        Ty::new(interner, TyKind::Infer(infer))
    }

    pub fn new_int_var(interner: DbInterner<'db>, v: IntVid) -> Self {
        Ty::new_infer(interner, InferTy::IntVar(v))
    }

    pub fn new_float_var(interner: DbInterner<'db>, v: FloatVid) -> Self {
        Ty::new_infer(interner, InferTy::FloatVar(v))
    }

    #[inline]
    pub fn new_int(interner: DbInterner<'db>, i: IntTy) -> Self {
        let types = interner.default_types();
        match i {
            IntTy::Isize => types.types.isize,
            IntTy::I8 => types.types.i8,
            IntTy::I16 => types.types.i16,
            IntTy::I32 => types.types.i32,
            IntTy::I64 => types.types.i64,
            IntTy::I128 => types.types.i128,
        }
    }

    pub fn new_uint(interner: DbInterner<'db>, ui: UintTy) -> Self {
        let types = interner.default_types();
        match ui {
            UintTy::Usize => types.types.usize,
            UintTy::U8 => types.types.u8,
            UintTy::U16 => types.types.u16,
            UintTy::U32 => types.types.u32,
            UintTy::U64 => types.types.u64,
            UintTy::U128 => types.types.u128,
        }
    }

    pub fn new_float(interner: DbInterner<'db>, f: FloatTy) -> Self {
        let types = interner.default_types();
        match f {
            FloatTy::F16 => types.types.f16,
            FloatTy::F32 => types.types.f32,
            FloatTy::F64 => types.types.f64,
            FloatTy::F128 => types.types.f128,
        }
    }

    pub fn new_fresh(interner: DbInterner<'db>, n: u32) -> Self {
        Ty::new_infer(interner, InferTy::FreshTy(n))
    }

    pub fn new_fresh_int(interner: DbInterner<'db>, n: u32) -> Self {
        Ty::new_infer(interner, InferTy::FreshIntTy(n))
    }

    pub fn new_fresh_float(interner: DbInterner<'db>, n: u32) -> Self {
        Ty::new_infer(interner, InferTy::FreshFloatTy(n))
    }

    pub fn new_empty_tuple(interner: DbInterner<'db>) -> Self {
        interner.default_types().types.unit
    }

    pub fn new_imm_ptr(interner: DbInterner<'db>, ty: Ty<'db>) -> Self {
        Ty::new_ptr(interner, ty, Mutability::Not)
    }

    pub fn new_imm_ref(interner: DbInterner<'db>, region: Region<'db>, ty: Ty<'db>) -> Self {
        Ty::new_ref(interner, region, ty, Mutability::Not)
    }

    pub fn new_opaque(
        interner: DbInterner<'db>,
        def_id: SolverDefId,
        args: GenericArgs<'db>,
    ) -> Self {
        Ty::new_alias(interner, AliasTyKind::Opaque, AliasTy::new_from_args(interner, def_id, args))
    }

    /// Returns the `Size` for primitive types (bool, uint, int, char, float).
    pub fn primitive_size(self, interner: DbInterner<'db>) -> Size {
        match self.kind() {
            TyKind::Bool => Size::from_bytes(1),
            TyKind::Char => Size::from_bytes(4),
            TyKind::Int(ity) => Integer::from_int_ty(&interner, ity).size(),
            TyKind::Uint(uty) => Integer::from_uint_ty(&interner, uty).size(),
            TyKind::Float(fty) => Float::from_float_ty(fty).size(),
            _ => panic!("non primitive type"),
        }
    }

    pub fn int_size_and_signed(self, interner: DbInterner<'db>) -> (Size, bool) {
        match self.kind() {
            TyKind::Int(ity) => (Integer::from_int_ty(&interner, ity).size(), true),
            TyKind::Uint(uty) => (Integer::from_uint_ty(&interner, uty).size(), false),
            _ => panic!("non integer discriminant"),
        }
    }

    pub fn walk(self) -> TypeWalker<DbInterner<'db>> {
        TypeWalker::new(self.into())
    }

    /// Fast path helper for testing if a type is `Sized` or `MetaSized`.
    ///
    /// Returning true means the type is known to implement the sizedness trait. Returning `false`
    /// means nothing -- could be sized, might not be.
    ///
    /// Note that we could never rely on the fact that a type such as `[_]` is trivially `!Sized`
    /// because we could be in a type environment with a bound such as `[_]: Copy`. A function with
    /// such a bound obviously never can be called, but that doesn't mean it shouldn't typecheck.
    /// This is why this method doesn't return `Option<bool>`.
    #[tracing::instrument(skip(tcx), level = "debug")]
    pub fn has_trivial_sizedness(self, tcx: DbInterner<'db>, sizedness: SizedTraitKind) -> bool {
        match self.kind() {
            TyKind::Infer(InferTy::IntVar(_) | InferTy::FloatVar(_))
            | TyKind::Uint(_)
            | TyKind::Int(_)
            | TyKind::Bool
            | TyKind::Float(_)
            | TyKind::FnDef(..)
            | TyKind::FnPtr(..)
            | TyKind::UnsafeBinder(_)
            | TyKind::RawPtr(..)
            | TyKind::Char
            | TyKind::Ref(..)
            | TyKind::Coroutine(..)
            | TyKind::CoroutineWitness(..)
            | TyKind::Array(..)
            | TyKind::Pat(..)
            | TyKind::Closure(..)
            | TyKind::CoroutineClosure(..)
            | TyKind::Never
            | TyKind::Error(_) => true,

            TyKind::Str | TyKind::Slice(_) | TyKind::Dynamic(_, _) => match sizedness {
                SizedTraitKind::Sized => false,
                SizedTraitKind::MetaSized => true,
            },

            TyKind::Foreign(..) => match sizedness {
                SizedTraitKind::Sized | SizedTraitKind::MetaSized => false,
            },

            TyKind::Tuple(tys) => {
                tys.last().is_none_or(|ty| ty.has_trivial_sizedness(tcx, sizedness))
            }

            TyKind::Adt(def, args) => def
                .sizedness_constraint(tcx, sizedness)
                .is_none_or(|ty| ty.instantiate(tcx, args).has_trivial_sizedness(tcx, sizedness)),

            TyKind::Alias(..) | TyKind::Param(_) | TyKind::Placeholder(..) | TyKind::Bound(..) => {
                false
            }

            TyKind::Infer(InferTy::TyVar(_)) => false,

            TyKind::Infer(
                InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_),
            ) => {
                panic!("`has_trivial_sizedness` applied to unexpected type: {self:?}")
            }
        }
    }

    /// Fast path helper for primitives which are always `Copy` and which
    /// have a side-effect-free `Clone` impl.
    ///
    /// Returning true means the type is known to be pure and `Copy+Clone`.
    /// Returning `false` means nothing -- could be `Copy`, might not be.
    ///
    /// This is mostly useful for optimizations, as these are the types
    /// on which we can replace cloning with dereferencing.
    pub fn is_trivially_pure_clone_copy(self) -> bool {
        match self.kind() {
            TyKind::Bool | TyKind::Char | TyKind::Never => true,

            // These aren't even `Clone`
            TyKind::Str | TyKind::Slice(..) | TyKind::Foreign(..) | TyKind::Dynamic(..) => false,

            TyKind::Infer(InferTy::FloatVar(_) | InferTy::IntVar(_))
            | TyKind::Int(..)
            | TyKind::Uint(..)
            | TyKind::Float(..) => true,

            // ZST which can't be named are fine.
            TyKind::FnDef(..) => true,

            TyKind::Array(element_ty, _len) => element_ty.is_trivially_pure_clone_copy(),

            // A 100-tuple isn't "trivial", so doing this only for reasonable sizes.
            TyKind::Tuple(field_tys) => {
                field_tys.len() <= 3 && field_tys.iter().all(Self::is_trivially_pure_clone_copy)
            }

            TyKind::Pat(ty, _) => ty.is_trivially_pure_clone_copy(),

            // Sometimes traits aren't implemented for every ABI or arity,
            // because we can't be generic over everything yet.
            TyKind::FnPtr(..) => false,

            // Definitely absolutely not copy.
            TyKind::Ref(_, _, Mutability::Mut) => false,

            // The standard library has a blanket Copy impl for shared references and raw pointers,
            // for all unsized types.
            TyKind::Ref(_, _, Mutability::Not) | TyKind::RawPtr(..) => true,

            TyKind::Coroutine(..) | TyKind::CoroutineWitness(..) => false,

            // Might be, but not "trivial" so just giving the safe answer.
            TyKind::Adt(..) | TyKind::Closure(..) | TyKind::CoroutineClosure(..) => false,

            TyKind::UnsafeBinder(_) => false,

            // Needs normalization or revealing to determine, so no is the safe answer.
            TyKind::Alias(..) => false,

            TyKind::Param(..)
            | TyKind::Placeholder(..)
            | TyKind::Bound(..)
            | TyKind::Infer(..)
            | TyKind::Error(..) => false,
        }
    }

    pub fn is_trivially_wf(self, tcx: DbInterner<'db>) -> bool {
        match self.kind() {
            TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Str
            | TyKind::Never
            | TyKind::Param(_)
            | TyKind::Placeholder(_)
            | TyKind::Bound(..) => true,

            TyKind::Slice(ty) => {
                ty.is_trivially_wf(tcx) && ty.has_trivial_sizedness(tcx, SizedTraitKind::Sized)
            }
            TyKind::RawPtr(ty, _) => ty.is_trivially_wf(tcx),

            TyKind::FnPtr(sig_tys, _) => {
                sig_tys.skip_binder().inputs_and_output.iter().all(|ty| ty.is_trivially_wf(tcx))
            }
            TyKind::Ref(_, ty, _) => ty.is_global() && ty.is_trivially_wf(tcx),

            TyKind::Infer(infer) => match infer {
                InferTy::TyVar(_) => false,
                InferTy::IntVar(_) | InferTy::FloatVar(_) => true,
                InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_) => true,
            },

            TyKind::Adt(_, _)
            | TyKind::Tuple(_)
            | TyKind::Array(..)
            | TyKind::Foreign(_)
            | TyKind::Pat(_, _)
            | TyKind::FnDef(..)
            | TyKind::UnsafeBinder(..)
            | TyKind::Dynamic(..)
            | TyKind::Closure(..)
            | TyKind::CoroutineClosure(..)
            | TyKind::Coroutine(..)
            | TyKind::CoroutineWitness(..)
            | TyKind::Alias(..)
            | TyKind::Error(_) => false,
        }
    }

    #[inline]
    pub fn is_never(self) -> bool {
        matches!(self.kind(), TyKind::Never)
    }

    #[inline]
    pub fn is_bool(self) -> bool {
        matches!(self.kind(), TyKind::Bool)
    }

    #[inline]
    pub fn is_char(self) -> bool {
        matches!(self.kind(), TyKind::Char)
    }

    /// A scalar type is one that denotes an atomic datum, with no sub-components.
    /// (A RawPtr is scalar because it represents a non-managed pointer, so its
    /// contents are abstract to rustc.)
    #[inline]
    pub fn is_scalar(self) -> bool {
        matches!(
            self.kind(),
            TyKind::Bool
                | TyKind::Char
                | TyKind::Int(_)
                | TyKind::Float(_)
                | TyKind::Uint(_)
                | TyKind::FnDef(..)
                | TyKind::FnPtr(..)
                | TyKind::RawPtr(_, _)
                | TyKind::Infer(InferTy::IntVar(_) | InferTy::FloatVar(_))
        )
    }

    #[inline]
    pub fn is_infer(self) -> bool {
        matches!(self.kind(), TyKind::Infer(..))
    }

    #[inline]
    pub fn is_numeric(self) -> bool {
        self.is_integral() || self.is_floating_point()
    }

    #[inline]
    pub fn is_str(self) -> bool {
        matches!(self.kind(), TyKind::Str)
    }

    #[inline]
    pub fn is_unit(self) -> bool {
        matches!(self.kind(), TyKind::Tuple(tys) if tys.is_empty())
    }

    #[inline]
    pub fn is_u8(self) -> bool {
        matches!(self.kind(), TyKind::Uint(UintTy::U8))
    }

    #[inline]
    pub fn is_raw_ptr(self) -> bool {
        matches!(self.kind(), TyKind::RawPtr(..))
    }

    #[inline]
    pub fn is_array(self) -> bool {
        matches!(self.kind(), TyKind::Array(..))
    }

    #[inline]
    pub fn is_slice(self) -> bool {
        matches!(self.kind(), TyKind::Slice(..))
    }

    pub fn is_union(self) -> bool {
        self.as_adt().is_some_and(|(adt, _)| matches!(adt, AdtId::UnionId(_)))
    }

    pub fn boxed_ty(self) -> Option<Ty<'db>> {
        match self.kind() {
            TyKind::Adt(adt_def, args) if adt_def.is_box() => Some(args.type_at(0)),
            _ => None,
        }
    }

    #[inline]
    pub fn as_adt(self) -> Option<(AdtId, GenericArgs<'db>)> {
        match self.kind() {
            TyKind::Adt(adt_def, args) => Some((adt_def.def_id().0, args)),
            _ => None,
        }
    }

    #[inline]
    pub fn as_slice(self) -> Option<Ty<'db>> {
        match self.kind() {
            TyKind::Slice(ty) => Some(ty),
            _ => None,
        }
    }

    #[inline]
    pub fn ty_vid(self) -> Option<TyVid> {
        match self.kind() {
            TyKind::Infer(rustc_type_ir::TyVar(vid)) => Some(vid),
            _ => None,
        }
    }

    /// Given a `fn` type, returns an equivalent `unsafe fn` type;
    /// that is, a `fn` type that is equivalent in every way for being
    /// unsafe.
    pub fn safe_to_unsafe_fn_ty(interner: DbInterner<'db>, sig: PolyFnSig<'db>) -> Ty<'db> {
        assert!(sig.safety().is_safe());
        Ty::new_fn_ptr(interner, sig.map_bound(|sig| FnSig { safety: Safety::Unsafe, ..sig }))
    }

    /// Returns the type of `*ty`.
    ///
    /// The parameter `explicit` indicates if this is an *explicit* dereference.
    /// Some types -- notably raw ptrs -- can only be dereferenced explicitly.
    pub fn builtin_deref(self, explicit: bool) -> Option<Ty<'db>> {
        match self.kind() {
            TyKind::Adt(adt, substs) if adt.is_box() => Some(substs.as_slice()[0].expect_ty()),
            TyKind::Ref(_, ty, _) => Some(ty),
            TyKind::RawPtr(ty, _) if explicit => Some(ty),
            _ => None,
        }
    }

    /// Whether the type contains some non-lifetime, aka. type or const, error type.
    pub fn references_non_lt_error(self) -> bool {
        references_non_lt_error(&self)
    }

    /// Whether the type contains a type error (ignoring const and lifetime errors).
    pub fn references_only_ty_error(self) -> bool {
        references_only_ty_error(&self)
    }

    pub fn callable_sig(self, interner: DbInterner<'db>) -> Option<Binder<'db, FnSig<'db>>> {
        match self.kind() {
            TyKind::FnDef(callable, args) => {
                Some(interner.fn_sig(callable).instantiate(interner, args))
            }
            TyKind::FnPtr(sig, hdr) => Some(sig.with(hdr)),
            TyKind::Closure(_, closure_args) => {
                Some(interner.signature_unclosure(closure_args.as_closure().sig(), Safety::Safe))
            }
            TyKind::CoroutineClosure(coroutine_id, args) => {
                Some(args.as_coroutine_closure().coroutine_closure_sig().map_bound(|sig| {
                    let unit_ty = Ty::new_unit(interner);
                    let return_ty = Ty::new_coroutine(
                        interner,
                        coroutine_id,
                        CoroutineArgs::new(
                            interner,
                            CoroutineArgsParts {
                                parent_args: args.as_coroutine_closure().parent_args(),
                                kind_ty: unit_ty,
                                resume_ty: unit_ty,
                                yield_ty: unit_ty,
                                return_ty: sig.return_ty,
                                // FIXME: Deduce this from the coroutine closure's upvars.
                                tupled_upvars_ty: unit_ty,
                            },
                        )
                        .args,
                    );
                    FnSig {
                        inputs_and_output: Tys::new_from_iter(
                            interner,
                            sig.tupled_inputs_ty
                                .tuple_fields()
                                .iter()
                                .chain(std::iter::once(return_ty)),
                        ),
                        c_variadic: sig.c_variadic,
                        safety: sig.safety,
                        abi: sig.abi,
                    }
                }))
            }
            _ => None,
        }
    }

    pub fn as_reference(self) -> Option<(Ty<'db>, Region<'db>, Mutability)> {
        match self.kind() {
            TyKind::Ref(region, ty, mutability) => Some((ty, region, mutability)),
            _ => None,
        }
    }

    pub fn as_reference_or_ptr(self) -> Option<(Ty<'db>, Rawness, Mutability)> {
        match self.kind() {
            TyKind::Ref(_, ty, mutability) => Some((ty, Rawness::Ref, mutability)),
            TyKind::RawPtr(ty, mutability) => Some((ty, Rawness::RawPtr, mutability)),
            _ => None,
        }
    }

    pub fn as_tuple(self) -> Option<Tys<'db>> {
        match self.kind() {
            TyKind::Tuple(tys) => Some(tys),
            _ => None,
        }
    }

    pub fn dyn_trait(self) -> Option<TraitId> {
        let TyKind::Dynamic(bounds, _) = self.kind() else { return None };
        Some(bounds.principal_def_id()?.0)
    }

    pub fn strip_references(self) -> Ty<'db> {
        let mut t = self;
        while let TyKind::Ref(_lifetime, ty, _mutability) = t.kind() {
            t = ty;
        }
        t
    }

    pub fn strip_reference(self) -> Ty<'db> {
        self.as_reference().map_or(self, |(ty, _, _)| ty)
    }

    /// Replace infer vars with errors.
    ///
    /// This needs to be called for every type that may contain infer vars and is yielded to outside inference,
    /// as things other than inference do not expect to see infer vars.
    pub fn replace_infer_with_error(self, interner: DbInterner<'db>) -> Ty<'db> {
        self.fold_with(&mut crate::next_solver::infer::resolve::ReplaceInferWithError::new(
            interner,
        ))
    }

    pub fn from_builtin_type(
        interner: DbInterner<'db>,
        ty: hir_def::builtin_type::BuiltinType,
    ) -> Ty<'db> {
        let types = interner.default_types();
        match ty {
            hir_def::builtin_type::BuiltinType::Char => types.types.char,
            hir_def::builtin_type::BuiltinType::Bool => types.types.bool,
            hir_def::builtin_type::BuiltinType::Str => types.types.str,
            hir_def::builtin_type::BuiltinType::Int(int) => match int {
                hir_def::builtin_type::BuiltinInt::Isize => types.types.isize,
                hir_def::builtin_type::BuiltinInt::I8 => types.types.i8,
                hir_def::builtin_type::BuiltinInt::I16 => types.types.i16,
                hir_def::builtin_type::BuiltinInt::I32 => types.types.i32,
                hir_def::builtin_type::BuiltinInt::I64 => types.types.i64,
                hir_def::builtin_type::BuiltinInt::I128 => types.types.i128,
            },
            hir_def::builtin_type::BuiltinType::Uint(uint) => match uint {
                hir_def::builtin_type::BuiltinUint::Usize => types.types.usize,
                hir_def::builtin_type::BuiltinUint::U8 => types.types.u8,
                hir_def::builtin_type::BuiltinUint::U16 => types.types.u16,
                hir_def::builtin_type::BuiltinUint::U32 => types.types.u32,
                hir_def::builtin_type::BuiltinUint::U64 => types.types.u64,
                hir_def::builtin_type::BuiltinUint::U128 => types.types.u128,
            },
            hir_def::builtin_type::BuiltinType::Float(float) => match float {
                hir_def::builtin_type::BuiltinFloat::F16 => types.types.f16,
                hir_def::builtin_type::BuiltinFloat::F32 => types.types.f32,
                hir_def::builtin_type::BuiltinFloat::F64 => types.types.f64,
                hir_def::builtin_type::BuiltinFloat::F128 => types.types.f128,
            },
        }
    }

    pub fn as_builtin(self) -> Option<hir_def::builtin_type::BuiltinType> {
        let builtin = match self.kind() {
            TyKind::Char => hir_def::builtin_type::BuiltinType::Char,
            TyKind::Bool => hir_def::builtin_type::BuiltinType::Bool,
            TyKind::Str => hir_def::builtin_type::BuiltinType::Str,
            TyKind::Int(int) => hir_def::builtin_type::BuiltinType::Int(match int {
                rustc_type_ir::IntTy::Isize => hir_def::builtin_type::BuiltinInt::Isize,
                rustc_type_ir::IntTy::I8 => hir_def::builtin_type::BuiltinInt::I8,
                rustc_type_ir::IntTy::I16 => hir_def::builtin_type::BuiltinInt::I16,
                rustc_type_ir::IntTy::I32 => hir_def::builtin_type::BuiltinInt::I32,
                rustc_type_ir::IntTy::I64 => hir_def::builtin_type::BuiltinInt::I64,
                rustc_type_ir::IntTy::I128 => hir_def::builtin_type::BuiltinInt::I128,
            }),
            TyKind::Uint(uint) => hir_def::builtin_type::BuiltinType::Uint(match uint {
                rustc_type_ir::UintTy::Usize => hir_def::builtin_type::BuiltinUint::Usize,
                rustc_type_ir::UintTy::U8 => hir_def::builtin_type::BuiltinUint::U8,
                rustc_type_ir::UintTy::U16 => hir_def::builtin_type::BuiltinUint::U16,
                rustc_type_ir::UintTy::U32 => hir_def::builtin_type::BuiltinUint::U32,
                rustc_type_ir::UintTy::U64 => hir_def::builtin_type::BuiltinUint::U64,
                rustc_type_ir::UintTy::U128 => hir_def::builtin_type::BuiltinUint::U128,
            }),
            TyKind::Float(float) => hir_def::builtin_type::BuiltinType::Float(match float {
                rustc_type_ir::FloatTy::F16 => hir_def::builtin_type::BuiltinFloat::F16,
                rustc_type_ir::FloatTy::F32 => hir_def::builtin_type::BuiltinFloat::F32,
                rustc_type_ir::FloatTy::F64 => hir_def::builtin_type::BuiltinFloat::F64,
                rustc_type_ir::FloatTy::F128 => hir_def::builtin_type::BuiltinFloat::F128,
            }),
            _ => return None,
        };
        Some(builtin)
    }

    // FIXME: Should this be here?
    pub fn impl_trait_bounds(self, db: &'db dyn HirDatabase) -> Option<Vec<Clause<'db>>> {
        let interner = DbInterner::new_no_crate(db);

        match self.kind() {
            TyKind::Alias(AliasTyKind::Opaque, opaque_ty) => Some(
                opaque_ty
                    .def_id
                    .expect_opaque_ty()
                    .predicates(db)
                    .iter_instantiated_copied(interner, opaque_ty.args.as_slice())
                    .collect(),
            ),
            TyKind::Param(param) => {
                // FIXME: We shouldn't use `param.id` here.
                let generic_params = db.generic_params(param.id.parent());
                let param_data = &generic_params[param.id.local_id()];
                match param_data {
                    TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                        TypeParamProvenance::ArgumentImplTrait => {
                            let predicates = GenericPredicates::query_all(db, param.id.parent())
                                .iter_identity_copied()
                                .filter(|wc| match wc.kind().skip_binder() {
                                    ClauseKind::Trait(tr) => tr.self_ty() == self,
                                    ClauseKind::Projection(pred) => pred.self_ty() == self,
                                    ClauseKind::TypeOutlives(pred) => pred.0 == self,
                                    _ => false,
                                })
                                .collect::<Vec<_>>();

                            Some(predicates)
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }
            TyKind::Coroutine(coroutine_id, _args) => {
                let InternedCoroutine(owner, _) = coroutine_id.0.loc(db);
                let krate = owner.module(db).krate(db);
                if let Some(future_trait) = hir_def::lang_item::lang_items(db, krate).Future {
                    // This is only used by type walking.
                    // Parameters will be walked outside, and projection predicate is not used.
                    // So just provide the Future trait.
                    let impl_bound = TraitRef::new_from_args(
                        interner,
                        future_trait.into(),
                        GenericArgs::empty(interner),
                    )
                    .upcast(interner);
                    Some(vec![impl_bound])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// FIXME: Get rid of this, it's not a good abstraction
    pub fn equals_ctor(self, other: Ty<'db>) -> bool {
        match (self.kind(), other.kind()) {
            (TyKind::Adt(adt, ..), TyKind::Adt(adt2, ..)) => adt.def_id() == adt2.def_id(),
            (TyKind::Slice(_), TyKind::Slice(_)) | (TyKind::Array(_, _), TyKind::Array(_, _)) => {
                true
            }
            (TyKind::FnDef(def_id, ..), TyKind::FnDef(def_id2, ..)) => def_id == def_id2,
            (TyKind::Alias(_, alias, ..), TyKind::Alias(_, alias2)) => {
                alias.def_id == alias2.def_id
            }
            (TyKind::Foreign(ty_id, ..), TyKind::Foreign(ty_id2, ..)) => ty_id == ty_id2,
            (TyKind::Closure(id1, _), TyKind::Closure(id2, _)) => id1 == id2,
            (TyKind::Ref(.., mutability), TyKind::Ref(.., mutability2))
            | (TyKind::RawPtr(.., mutability), TyKind::RawPtr(.., mutability2)) => {
                mutability == mutability2
            }
            (TyKind::FnPtr(sig, hdr), TyKind::FnPtr(sig2, hdr2)) => sig == sig2 && hdr == hdr2,
            (TyKind::Tuple(tys), TyKind::Tuple(tys2)) => tys.len() == tys2.len(),
            (TyKind::Str, TyKind::Str)
            | (TyKind::Never, TyKind::Never)
            | (TyKind::Char, TyKind::Char)
            | (TyKind::Bool, TyKind::Bool) => true,
            (TyKind::Int(int), TyKind::Int(int2)) => int == int2,
            (TyKind::Float(float), TyKind::Float(float2)) => float == float2,
            _ => false,
        }
    }
}

pub fn references_non_lt_error<'db, T: TypeVisitableExt<DbInterner<'db>>>(t: &T) -> bool {
    t.references_error() && t.visit_with(&mut ReferencesNonLifetimeError).is_break()
}

struct ReferencesNonLifetimeError;

impl<'db> TypeVisitor<DbInterner<'db>> for ReferencesNonLifetimeError {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'db>) -> Self::Result {
        if ty.is_ty_error() { ControlFlow::Break(()) } else { ty.super_visit_with(self) }
    }

    fn visit_const(&mut self, c: Const<'db>) -> Self::Result {
        if c.is_ct_error() { ControlFlow::Break(()) } else { c.super_visit_with(self) }
    }
}

pub fn references_only_ty_error<'db, T: TypeVisitableExt<DbInterner<'db>>>(t: &T) -> bool {
    t.references_error() && t.visit_with(&mut ReferencesOnlyTyError).is_break()
}

struct ReferencesOnlyTyError;

impl<'db> TypeVisitor<DbInterner<'db>> for ReferencesOnlyTyError {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, ty: Ty<'db>) -> Self::Result {
        if ty.is_ty_error() { ControlFlow::Break(()) } else { ty.super_visit_with(self) }
    }
}

impl<'db> std::fmt::Debug for Ty<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner().internee.fmt(f)
    }
}

impl<'db> IntoKind for Ty<'db> {
    type Kind = TyKind<'db>;

    #[inline]
    fn kind(self) -> Self::Kind {
        self.inner().internee
    }
}

impl<'db, V: super::WorldExposer> GenericTypeVisitable<V> for Ty<'db> {
    fn generic_visit_with(&self, visitor: &mut V) {
        if visitor.on_interned(self.interned).is_continue() {
            self.kind().generic_visit_with(visitor);
        }
    }
}

impl<'db> TypeVisitable<DbInterner<'db>> for Ty<'db> {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        visitor.visit_ty(*self)
    }
}

impl<'db> TypeSuperVisitable<DbInterner<'db>> for Ty<'db> {
    fn super_visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        match (*self).kind() {
            TyKind::RawPtr(ty, _mutbl) => ty.visit_with(visitor),
            TyKind::Array(typ, sz) => {
                try_visit!(typ.visit_with(visitor));
                sz.visit_with(visitor)
            }
            TyKind::Slice(typ) => typ.visit_with(visitor),
            TyKind::Adt(_, args) => args.visit_with(visitor),
            TyKind::Dynamic(ref trait_ty, ref reg) => {
                try_visit!(trait_ty.visit_with(visitor));
                reg.visit_with(visitor)
            }
            TyKind::Tuple(ts) => ts.visit_with(visitor),
            TyKind::FnDef(_, args) => args.visit_with(visitor),
            TyKind::FnPtr(ref sig_tys, _) => sig_tys.visit_with(visitor),
            TyKind::UnsafeBinder(f) => f.visit_with(visitor),
            TyKind::Ref(r, ty, _) => {
                try_visit!(r.visit_with(visitor));
                ty.visit_with(visitor)
            }
            TyKind::Coroutine(_did, ref args) => args.visit_with(visitor),
            TyKind::CoroutineWitness(_did, ref args) => args.visit_with(visitor),
            TyKind::Closure(_did, ref args) => args.visit_with(visitor),
            TyKind::CoroutineClosure(_did, ref args) => args.visit_with(visitor),
            TyKind::Alias(_, ref data) => data.visit_with(visitor),

            TyKind::Pat(ty, pat) => {
                try_visit!(ty.visit_with(visitor));
                pat.visit_with(visitor)
            }

            TyKind::Error(guar) => guar.visit_with(visitor),

            TyKind::Bool
            | TyKind::Char
            | TyKind::Str
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Infer(_)
            | TyKind::Bound(..)
            | TyKind::Placeholder(..)
            | TyKind::Param(..)
            | TyKind::Never
            | TyKind::Foreign(..) => V::Result::output(),
        }
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for Ty<'db> {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        folder.try_fold_ty(self)
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, folder: &mut F) -> Self {
        folder.fold_ty(self)
    }
}

impl<'db> TypeSuperFoldable<DbInterner<'db>> for Ty<'db> {
    fn try_super_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match self.kind() {
            TyKind::RawPtr(ty, mutbl) => TyKind::RawPtr(ty.try_fold_with(folder)?, mutbl),
            TyKind::Array(typ, sz) => {
                TyKind::Array(typ.try_fold_with(folder)?, sz.try_fold_with(folder)?)
            }
            TyKind::Slice(typ) => TyKind::Slice(typ.try_fold_with(folder)?),
            TyKind::Adt(tid, args) => TyKind::Adt(tid, args.try_fold_with(folder)?),
            TyKind::Dynamic(trait_ty, region) => {
                TyKind::Dynamic(trait_ty.try_fold_with(folder)?, region.try_fold_with(folder)?)
            }
            TyKind::Tuple(ts) => TyKind::Tuple(ts.try_fold_with(folder)?),
            TyKind::FnDef(def_id, args) => TyKind::FnDef(def_id, args.try_fold_with(folder)?),
            TyKind::FnPtr(sig_tys, hdr) => TyKind::FnPtr(sig_tys.try_fold_with(folder)?, hdr),
            TyKind::UnsafeBinder(f) => TyKind::UnsafeBinder(f.try_fold_with(folder)?),
            TyKind::Ref(r, ty, mutbl) => {
                TyKind::Ref(r.try_fold_with(folder)?, ty.try_fold_with(folder)?, mutbl)
            }
            TyKind::Coroutine(did, args) => TyKind::Coroutine(did, args.try_fold_with(folder)?),
            TyKind::CoroutineWitness(did, args) => {
                TyKind::CoroutineWitness(did, args.try_fold_with(folder)?)
            }
            TyKind::Closure(did, args) => TyKind::Closure(did, args.try_fold_with(folder)?),
            TyKind::CoroutineClosure(did, args) => {
                TyKind::CoroutineClosure(did, args.try_fold_with(folder)?)
            }
            TyKind::Alias(kind, data) => TyKind::Alias(kind, data.try_fold_with(folder)?),
            TyKind::Pat(ty, pat) => {
                TyKind::Pat(ty.try_fold_with(folder)?, pat.try_fold_with(folder)?)
            }

            TyKind::Bool
            | TyKind::Char
            | TyKind::Str
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Error(_)
            | TyKind::Infer(_)
            | TyKind::Param(..)
            | TyKind::Bound(..)
            | TyKind::Placeholder(..)
            | TyKind::Never
            | TyKind::Foreign(..) => return Ok(self),
        };

        Ok(if self.kind() == kind { self } else { Ty::new(folder.cx(), kind) })
    }
    fn super_fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(
        self,
        folder: &mut F,
    ) -> Self {
        let kind = match self.kind() {
            TyKind::RawPtr(ty, mutbl) => TyKind::RawPtr(ty.fold_with(folder), mutbl),
            TyKind::Array(typ, sz) => TyKind::Array(typ.fold_with(folder), sz.fold_with(folder)),
            TyKind::Slice(typ) => TyKind::Slice(typ.fold_with(folder)),
            TyKind::Adt(tid, args) => TyKind::Adt(tid, args.fold_with(folder)),
            TyKind::Dynamic(trait_ty, region) => {
                TyKind::Dynamic(trait_ty.fold_with(folder), region.fold_with(folder))
            }
            TyKind::Tuple(ts) => TyKind::Tuple(ts.fold_with(folder)),
            TyKind::FnDef(def_id, args) => TyKind::FnDef(def_id, args.fold_with(folder)),
            TyKind::FnPtr(sig_tys, hdr) => TyKind::FnPtr(sig_tys.fold_with(folder), hdr),
            TyKind::UnsafeBinder(f) => TyKind::UnsafeBinder(f.fold_with(folder)),
            TyKind::Ref(r, ty, mutbl) => {
                TyKind::Ref(r.fold_with(folder), ty.fold_with(folder), mutbl)
            }
            TyKind::Coroutine(did, args) => TyKind::Coroutine(did, args.fold_with(folder)),
            TyKind::CoroutineWitness(did, args) => {
                TyKind::CoroutineWitness(did, args.fold_with(folder))
            }
            TyKind::Closure(did, args) => TyKind::Closure(did, args.fold_with(folder)),
            TyKind::CoroutineClosure(did, args) => {
                TyKind::CoroutineClosure(did, args.fold_with(folder))
            }
            TyKind::Alias(kind, data) => TyKind::Alias(kind, data.fold_with(folder)),
            TyKind::Pat(ty, pat) => TyKind::Pat(ty.fold_with(folder), pat.fold_with(folder)),

            TyKind::Bool
            | TyKind::Char
            | TyKind::Str
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Error(_)
            | TyKind::Infer(_)
            | TyKind::Param(..)
            | TyKind::Bound(..)
            | TyKind::Placeholder(..)
            | TyKind::Never
            | TyKind::Foreign(..) => return self,
        };

        if self.kind() == kind { self } else { Ty::new(folder.cx(), kind) }
    }
}

impl<'db> Relate<DbInterner<'db>> for Ty<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        relation.tys(a, b)
    }
}

impl<'db> Flags for Ty<'db> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        self.inner().flags
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        self.inner().outer_exclusive_binder
    }
}

impl<'db> rustc_type_ir::inherent::Ty<DbInterner<'db>> for Ty<'db> {
    fn new_unit(interner: DbInterner<'db>) -> Self {
        interner.default_types().types.unit
    }

    fn new_bool(interner: DbInterner<'db>) -> Self {
        interner.default_types().types.bool
    }

    fn new_u8(interner: DbInterner<'db>) -> Self {
        interner.default_types().types.u8
    }

    fn new_usize(interner: DbInterner<'db>) -> Self {
        interner.default_types().types.usize
    }

    fn new_infer(interner: DbInterner<'db>, var: rustc_type_ir::InferTy) -> Self {
        Ty::new(interner, TyKind::Infer(var))
    }

    fn new_var(interner: DbInterner<'db>, var: rustc_type_ir::TyVid) -> Self {
        Ty::new(interner, TyKind::Infer(rustc_type_ir::InferTy::TyVar(var)))
    }

    fn new_param(interner: DbInterner<'db>, param: ParamTy) -> Self {
        Ty::new(interner, TyKind::Param(param))
    }

    fn new_placeholder(interner: DbInterner<'db>, param: PlaceholderTy) -> Self {
        Ty::new(interner, TyKind::Placeholder(param))
    }

    fn new_bound(interner: DbInterner<'db>, debruijn: DebruijnIndex, var: BoundTy) -> Self {
        Ty::new(interner, TyKind::Bound(BoundVarIndexKind::Bound(debruijn), var))
    }

    fn new_anon_bound(interner: DbInterner<'db>, debruijn: DebruijnIndex, var: BoundVar) -> Self {
        Ty::new(
            interner,
            TyKind::Bound(
                BoundVarIndexKind::Bound(debruijn),
                BoundTy { var, kind: BoundTyKind::Anon },
            ),
        )
    }

    fn new_canonical_bound(interner: DbInterner<'db>, var: BoundVar) -> Self {
        Ty::new(
            interner,
            TyKind::Bound(BoundVarIndexKind::Canonical, BoundTy { var, kind: BoundTyKind::Anon }),
        )
    }

    fn new_alias(interner: DbInterner<'db>, kind: AliasTyKind, alias_ty: AliasTy<'db>) -> Self {
        Ty::new(interner, TyKind::Alias(kind, alias_ty))
    }

    fn new_error(interner: DbInterner<'db>, guar: ErrorGuaranteed) -> Self {
        Ty::new(interner, TyKind::Error(guar))
    }

    fn new_adt(
        interner: DbInterner<'db>,
        adt_def: <DbInterner<'db> as Interner>::AdtDef,
        args: GenericArgs<'db>,
    ) -> Self {
        Ty::new(interner, TyKind::Adt(adt_def, args))
    }

    fn new_foreign(interner: DbInterner<'db>, def_id: TypeAliasIdWrapper) -> Self {
        Ty::new(interner, TyKind::Foreign(def_id))
    }

    fn new_dynamic(
        interner: DbInterner<'db>,
        preds: <DbInterner<'db> as Interner>::BoundExistentialPredicates,
        region: <DbInterner<'db> as Interner>::Region,
    ) -> Self {
        Ty::new(interner, TyKind::Dynamic(preds, region))
    }

    fn new_coroutine(
        interner: DbInterner<'db>,
        def_id: CoroutineIdWrapper,
        args: <DbInterner<'db> as Interner>::GenericArgs,
    ) -> Self {
        Ty::new(interner, TyKind::Coroutine(def_id, args))
    }

    fn new_coroutine_closure(
        interner: DbInterner<'db>,
        def_id: CoroutineIdWrapper,
        args: <DbInterner<'db> as Interner>::GenericArgs,
    ) -> Self {
        Ty::new(interner, TyKind::CoroutineClosure(def_id, args))
    }

    fn new_closure(
        interner: DbInterner<'db>,
        def_id: ClosureIdWrapper,
        args: <DbInterner<'db> as Interner>::GenericArgs,
    ) -> Self {
        Ty::new(interner, TyKind::Closure(def_id, args))
    }

    fn new_coroutine_witness(
        interner: DbInterner<'db>,
        def_id: CoroutineIdWrapper,
        args: <DbInterner<'db> as Interner>::GenericArgs,
    ) -> Self {
        Ty::new(interner, TyKind::CoroutineWitness(def_id, args))
    }

    fn new_coroutine_witness_for_coroutine(
        interner: DbInterner<'db>,
        def_id: CoroutineIdWrapper,
        coroutine_args: <DbInterner<'db> as Interner>::GenericArgs,
    ) -> Self {
        // HACK: Coroutine witness types are lifetime erased, so they
        // never reference any lifetime args from the coroutine. We erase
        // the regions here since we may get into situations where a
        // coroutine is recursively contained within itself, leading to
        // witness types that differ by region args. This means that
        // cycle detection in fulfillment will not kick in, which leads
        // to unnecessary overflows in async code. See the issue:
        // <https://github.com/rust-lang/rust/issues/145151>.
        let coroutine_args = interner.mk_args_from_iter(coroutine_args.iter().map(|arg| {
            match arg.kind() {
                GenericArgKind::Type(_) | GenericArgKind::Const(_) => arg,
                GenericArgKind::Lifetime(_) => {
                    crate::next_solver::Region::new(interner, rustc_type_ir::RegionKind::ReErased)
                        .into()
                }
            }
        }));
        Ty::new_coroutine_witness(interner, def_id, coroutine_args)
    }

    fn new_ptr(interner: DbInterner<'db>, ty: Self, mutbl: rustc_ast_ir::Mutability) -> Self {
        Ty::new(interner, TyKind::RawPtr(ty, mutbl))
    }

    fn new_ref(
        interner: DbInterner<'db>,
        region: <DbInterner<'db> as Interner>::Region,
        ty: Self,
        mutbl: rustc_ast_ir::Mutability,
    ) -> Self {
        Ty::new(interner, TyKind::Ref(region, ty, mutbl))
    }

    fn new_array_with_const_len(
        interner: DbInterner<'db>,
        ty: Self,
        len: <DbInterner<'db> as Interner>::Const,
    ) -> Self {
        Ty::new(interner, TyKind::Array(ty, len))
    }

    fn new_slice(interner: DbInterner<'db>, ty: Self) -> Self {
        Ty::new(interner, TyKind::Slice(ty))
    }

    fn new_tup(interner: DbInterner<'db>, tys: &[<DbInterner<'db> as Interner>::Ty]) -> Self {
        Ty::new(interner, TyKind::Tuple(Tys::new_from_slice(tys)))
    }

    fn new_tup_from_iter<It, T>(interner: DbInterner<'db>, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<Self, Self>,
    {
        T::collect_and_apply(iter, |ts| Ty::new_tup(interner, ts))
    }

    fn new_fn_def(
        interner: DbInterner<'db>,
        def_id: CallableIdWrapper,
        args: <DbInterner<'db> as Interner>::GenericArgs,
    ) -> Self {
        Ty::new(interner, TyKind::FnDef(def_id, args))
    }

    fn new_fn_ptr(
        interner: DbInterner<'db>,
        sig: rustc_type_ir::Binder<DbInterner<'db>, rustc_type_ir::FnSig<DbInterner<'db>>>,
    ) -> Self {
        let (sig_tys, header) = sig.split();
        Ty::new(interner, TyKind::FnPtr(sig_tys, header))
    }

    fn new_pat(
        interner: DbInterner<'db>,
        ty: Self,
        pat: <DbInterner<'db> as Interner>::Pat,
    ) -> Self {
        Ty::new(interner, TyKind::Pat(ty, pat))
    }

    fn new_unsafe_binder(
        interner: DbInterner<'db>,
        ty: rustc_type_ir::Binder<DbInterner<'db>, <DbInterner<'db> as Interner>::Ty>,
    ) -> Self {
        Ty::new(interner, TyKind::UnsafeBinder(ty.into()))
    }

    fn tuple_fields(self) -> <DbInterner<'db> as Interner>::Tys {
        match self.kind() {
            TyKind::Tuple(args) => args,
            _ => panic!("tuple_fields called on non-tuple: {self:?}"),
        }
    }

    fn to_opt_closure_kind(self) -> Option<rustc_type_ir::ClosureKind> {
        match self.kind() {
            TyKind::Int(int_ty) => match int_ty {
                IntTy::I8 => Some(ClosureKind::Fn),
                IntTy::I16 => Some(ClosureKind::FnMut),
                IntTy::I32 => Some(ClosureKind::FnOnce),
                _ => unreachable!("cannot convert type `{:?}` to a closure kind", self),
            },

            // "Bound" types appear in canonical queries when the
            // closure type is not yet known, and `Placeholder` and `Param`
            // may be encountered in generic `AsyncFnKindHelper` goals.
            TyKind::Bound(..) | TyKind::Placeholder(_) | TyKind::Param(_) | TyKind::Infer(_) => {
                None
            }

            TyKind::Error(_) => Some(ClosureKind::Fn),

            _ => unreachable!("cannot convert type `{:?}` to a closure kind", self),
        }
    }

    fn from_closure_kind(interner: DbInterner<'db>, kind: rustc_type_ir::ClosureKind) -> Self {
        let types = interner.default_types();
        match kind {
            ClosureKind::Fn => types.types.i8,
            ClosureKind::FnMut => types.types.i16,
            ClosureKind::FnOnce => types.types.i32,
        }
    }

    fn from_coroutine_closure_kind(
        interner: DbInterner<'db>,
        kind: rustc_type_ir::ClosureKind,
    ) -> Self {
        let types = interner.default_types();
        match kind {
            ClosureKind::Fn | ClosureKind::FnMut => types.types.i16,
            ClosureKind::FnOnce => types.types.i32,
        }
    }

    fn has_unsafe_fields(self) -> bool {
        false
    }

    fn discriminant_ty(self, interner: DbInterner<'db>) -> <DbInterner<'db> as Interner>::Ty {
        match self.kind() {
            TyKind::Adt(adt, _) if adt.is_enum() => adt.repr().discr_type().to_ty(interner),
            TyKind::Coroutine(_, args) => args.as_coroutine().discr_ty(interner),

            TyKind::Param(_) | TyKind::Alias(..) | TyKind::Infer(InferTy::TyVar(_)) => {
                /*
                let assoc_items = tcx.associated_item_def_ids(
                    tcx.require_lang_item(hir::LangItem::DiscriminantKind, None),
                );
                TyKind::new_projection_from_args(tcx, assoc_items[0], tcx.mk_args(&[self.into()]))
                */
                unimplemented!()
            }

            TyKind::Pat(ty, _) => ty.discriminant_ty(interner),

            TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Adt(..)
            | TyKind::Foreign(_)
            | TyKind::Str
            | TyKind::Array(..)
            | TyKind::Slice(_)
            | TyKind::RawPtr(_, _)
            | TyKind::Ref(..)
            | TyKind::FnDef(..)
            | TyKind::FnPtr(..)
            | TyKind::Dynamic(..)
            | TyKind::Closure(..)
            | TyKind::CoroutineClosure(..)
            | TyKind::CoroutineWitness(..)
            | TyKind::Never
            | TyKind::Tuple(_)
            | TyKind::Error(_)
            | TyKind::Infer(InferTy::IntVar(_) | InferTy::FloatVar(_)) => {
                interner.default_types().types.u8
            }

            TyKind::Bound(..)
            | TyKind::Placeholder(_)
            | TyKind::Infer(
                InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_),
            ) => {
                panic!(
                    "`dself.iter().map(|v| v.try_fold_with(folder)).collect::<Result<_, _>>()?iscriminant_ty` applied to unexpected type: {self:?}"
                )
            }
            TyKind::UnsafeBinder(..) => unimplemented!(),
        }
    }
}

interned_slice!(TysStorage, Tys, StoredTys, tys, Ty<'db>, Ty<'static>);
impl_foldable_for_interned_slice!(Tys);

impl<'db> Tys<'db> {
    #[inline]
    pub fn inputs(self) -> &'db [Ty<'db>] {
        self.as_slice().split_last().unwrap().1
    }
}

impl<'db> rustc_type_ir::inherent::Tys<DbInterner<'db>> for Tys<'db> {
    fn inputs(self) -> <DbInterner<'db> as Interner>::FnInputTys {
        self.as_slice().split_last().unwrap().1
    }

    fn output(self) -> <DbInterner<'db> as Interner>::Ty {
        *self.as_slice().split_last().unwrap().0
    }
}

pub type PlaceholderTy = Placeholder<BoundTy>;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ParamTy {
    // FIXME: I'm not pleased with this. Ideally a `Param` should only know its index - the defining item
    // is known from the `EarlyBinder`. This should also be beneficial for memory usage. But code currently
    // assumes it can get the definition from `Param` alone - so that's what we got.
    pub id: TypeParamId,
    pub index: u32,
}

impl ParamTy {
    pub fn to_ty<'db>(self, interner: DbInterner<'db>) -> Ty<'db> {
        Ty::new_param(interner, self.id, self.index)
    }
}

impl std::fmt::Debug for ParamTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct BoundTy {
    pub var: BoundVar,
    // FIXME: This is for diagnostics in rustc, do we really need it?
    pub kind: BoundTyKind,
}

impl std::fmt::Debug for BoundTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            BoundTyKind::Anon => write!(f, "{:?}", self.var),
            BoundTyKind::Param(def_id) => write!(f, "{def_id:?}"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BoundTyKind {
    Anon,
    Param(SolverDefId),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ErrorGuaranteed;

impl<V> GenericTypeVisitable<V> for ErrorGuaranteed {
    fn generic_visit_with(&self, _visitor: &mut V) {}
}

impl<'db> TypeVisitable<DbInterner<'db>> for ErrorGuaranteed {
    fn visit_with<V: rustc_type_ir::TypeVisitor<DbInterner<'db>>>(
        &self,
        visitor: &mut V,
    ) -> V::Result {
        visitor.visit_error(*self)
    }
}

impl<'db> TypeFoldable<DbInterner<'db>> for ErrorGuaranteed {
    fn try_fold_with<F: rustc_type_ir::FallibleTypeFolder<DbInterner<'db>>>(
        self,
        _folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self)
    }
    fn fold_with<F: rustc_type_ir::TypeFolder<DbInterner<'db>>>(self, _folder: &mut F) -> Self {
        self
    }
}

impl ParamLike for ParamTy {
    fn index(self) -> u32 {
        self.index
    }
}

impl<'db> BoundVarLike<DbInterner<'db>> for BoundTy {
    fn var(self) -> BoundVar {
        self.var
    }

    fn assert_eq(self, var: BoundVarKind) {
        assert_eq!(self.kind, var.expect_ty())
    }
}

impl<'db> PlaceholderLike<DbInterner<'db>> for PlaceholderTy {
    type Bound = BoundTy;

    fn universe(self) -> rustc_type_ir::UniverseIndex {
        self.universe
    }

    fn var(self) -> BoundVar {
        self.bound.var
    }

    fn with_updated_universe(self, ui: rustc_type_ir::UniverseIndex) -> Self {
        Placeholder { universe: ui, bound: self.bound }
    }

    fn new(ui: rustc_type_ir::UniverseIndex, bound: BoundTy) -> Self {
        Placeholder { universe: ui, bound }
    }

    fn new_anon(ui: rustc_type_ir::UniverseIndex, var: rustc_type_ir::BoundVar) -> Self {
        Placeholder { universe: ui, bound: BoundTy { var, kind: BoundTyKind::Anon } }
    }
}

impl<'db> DbInterner<'db> {
    /// Given a closure signature, returns an equivalent fn signature. Detuples
    /// and so forth -- so e.g., if we have a sig with `Fn<(u32, i32)>` then
    /// you would get a `fn(u32, i32)`.
    /// `unsafety` determines the unsafety of the fn signature. If you pass
    /// `Safety::Unsafe` in the previous example, then you would get
    /// an `unsafe fn (u32, i32)`.
    /// It cannot convert a closure that requires unsafe.
    pub fn signature_unclosure(self, sig: PolyFnSig<'db>, safety: Safety) -> PolyFnSig<'db> {
        sig.map_bound(|s| {
            let params = match s.inputs()[0].kind() {
                TyKind::Tuple(params) => params,
                _ => panic!(),
            };
            self.mk_fn_sig(params, s.output(), s.c_variadic, safety, FnAbi::Rust)
        })
    }
}
