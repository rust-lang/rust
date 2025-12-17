//! Various utilities for the next-trait-solver.

use std::ops::ControlFlow;

use hir_def::TraitId;
use rustc_abi::{Float, HasDataLayout, Integer, IntegerType, Primitive, ReprOptions};
use rustc_type_ir::{
    ConstKind, CoroutineArgs, DebruijnIndex, FloatTy, INNERMOST, IntTy, Interner,
    PredicatePolarity, RegionKind, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeSuperVisitable,
    TypeVisitableExt, TypeVisitor, UintTy, UniverseIndex, elaborate,
    inherent::{AdtDef, GenericArg as _, IntoKind, ParamEnv as _, SliceLike, Ty as _},
    lang_items::SolverTraitLangItem,
    solve::SizedTraitKind,
};

use crate::next_solver::{
    BoundConst, FxIndexMap, ParamEnv, Placeholder, PlaceholderConst, PlaceholderRegion,
    PolyTraitRef,
    infer::{
        InferCtxt,
        traits::{Obligation, ObligationCause, PredicateObligation},
    },
};

use super::{
    Binder, BoundRegion, BoundTy, Clause, ClauseKind, Const, DbInterner, EarlyBinder, GenericArgs,
    Predicate, PredicateKind, Region, SolverDefId, Ty, TyKind,
    fold::{BoundVarReplacer, FnMutDelegate},
};

#[derive(Clone, Debug)]
pub struct Discr<'db> {
    /// Bit representation of the discriminant (e.g., `-128i8` is `0xFF_u128`).
    pub val: u128,
    pub ty: Ty<'db>,
}

impl<'db> Discr<'db> {
    /// Adds `1` to the value and wraps around if the maximum for the type is reached.
    pub fn wrap_incr(self, interner: DbInterner<'db>) -> Self {
        self.checked_add(interner, 1).0
    }
    pub fn checked_add(self, interner: DbInterner<'db>, n: u128) -> (Self, bool) {
        let (size, signed) = self.ty.int_size_and_signed(interner);
        let (val, oflo) = if signed {
            let min = size.signed_int_min();
            let max = size.signed_int_max();
            let val = size.sign_extend(self.val);
            assert!(n < (i128::MAX as u128));
            let n = n as i128;
            let oflo = val > max - n;
            let val = if oflo { min + (n - (max - val) - 1) } else { val + n };
            // zero the upper bits
            let val = val as u128;
            let val = size.truncate(val);
            (val, oflo)
        } else {
            let max = size.unsigned_int_max();
            let val = self.val;
            let oflo = val > max - n;
            let val = if oflo { n - (max - val) - 1 } else { val + n };
            (val, oflo)
        };
        (Self { val, ty: self.ty }, oflo)
    }
}

pub trait IntegerTypeExt {
    fn to_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db>;
    fn initial_discriminant<'db>(&self, interner: DbInterner<'db>) -> Discr<'db>;
    fn disr_incr<'db>(
        &self,
        interner: DbInterner<'db>,
        val: Option<Discr<'db>>,
    ) -> Option<Discr<'db>>;
}

impl IntegerTypeExt for IntegerType {
    fn to_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db> {
        match self {
            IntegerType::Pointer(true) => Ty::new(interner, TyKind::Int(IntTy::Isize)),
            IntegerType::Pointer(false) => Ty::new(interner, TyKind::Uint(UintTy::Usize)),
            IntegerType::Fixed(i, s) => i.to_ty(interner, *s),
        }
    }

    fn initial_discriminant<'db>(&self, interner: DbInterner<'db>) -> Discr<'db> {
        Discr { val: 0, ty: self.to_ty(interner) }
    }

    fn disr_incr<'db>(
        &self,
        interner: DbInterner<'db>,
        val: Option<Discr<'db>>,
    ) -> Option<Discr<'db>> {
        if let Some(val) = val {
            assert_eq!(self.to_ty(interner), val.ty);
            let (new, oflo) = val.checked_add(interner, 1);
            if oflo { None } else { Some(new) }
        } else {
            Some(self.initial_discriminant(interner))
        }
    }
}

pub trait IntegerExt {
    fn to_ty<'db>(&self, interner: DbInterner<'db>, signed: bool) -> Ty<'db>;
    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: IntTy) -> Integer;
    fn from_uint_ty<C: HasDataLayout>(cx: &C, ity: UintTy) -> Integer;
    fn repr_discr<'db>(
        interner: DbInterner<'db>,
        ty: Ty<'db>,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (Integer, bool);
}

impl IntegerExt for Integer {
    #[inline]
    fn to_ty<'db>(&self, interner: DbInterner<'db>, signed: bool) -> Ty<'db> {
        use Integer::*;
        match (*self, signed) {
            (I8, false) => Ty::new(interner, TyKind::Uint(UintTy::U8)),
            (I16, false) => Ty::new(interner, TyKind::Uint(UintTy::U16)),
            (I32, false) => Ty::new(interner, TyKind::Uint(UintTy::U32)),
            (I64, false) => Ty::new(interner, TyKind::Uint(UintTy::U64)),
            (I128, false) => Ty::new(interner, TyKind::Uint(UintTy::U128)),
            (I8, true) => Ty::new(interner, TyKind::Int(IntTy::I8)),
            (I16, true) => Ty::new(interner, TyKind::Int(IntTy::I16)),
            (I32, true) => Ty::new(interner, TyKind::Int(IntTy::I32)),
            (I64, true) => Ty::new(interner, TyKind::Int(IntTy::I64)),
            (I128, true) => Ty::new(interner, TyKind::Int(IntTy::I128)),
        }
    }

    fn from_int_ty<C: HasDataLayout>(cx: &C, ity: IntTy) -> Integer {
        use Integer::*;
        match ity {
            IntTy::I8 => I8,
            IntTy::I16 => I16,
            IntTy::I32 => I32,
            IntTy::I64 => I64,
            IntTy::I128 => I128,
            IntTy::Isize => cx.data_layout().ptr_sized_integer(),
        }
    }
    fn from_uint_ty<C: HasDataLayout>(cx: &C, ity: UintTy) -> Integer {
        use Integer::*;
        match ity {
            UintTy::U8 => I8,
            UintTy::U16 => I16,
            UintTy::U32 => I32,
            UintTy::U64 => I64,
            UintTy::U128 => I128,
            UintTy::Usize => cx.data_layout().ptr_sized_integer(),
        }
    }

    /// Finds the appropriate Integer type and signedness for the given
    /// signed discriminant range and `#[repr]` attribute.
    /// N.B.: `u128` values above `i128::MAX` will be treated as signed, but
    /// that shouldn't affect anything, other than maybe debuginfo.
    fn repr_discr<'db>(
        interner: DbInterner<'db>,
        ty: Ty<'db>,
        repr: &ReprOptions,
        min: i128,
        max: i128,
    ) -> (Integer, bool) {
        // Theoretically, negative values could be larger in unsigned representation
        // than the unsigned representation of the signed minimum. However, if there
        // are any negative values, the only valid unsigned representation is u128
        // which can fit all i128 values, so the result remains unaffected.
        let unsigned_fit = Integer::fit_unsigned(std::cmp::max(min as u128, max as u128));
        let signed_fit = std::cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

        if let Some(ity) = repr.int {
            let discr = Integer::from_attr(&interner, ity);
            let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
            if discr < fit {
                panic!(
                    "Integer::repr_discr: `#[repr]` hint too small for \
                      discriminant range of enum `{ty:?}`"
                )
            }
            return (discr, ity.is_signed());
        }

        let at_least = if repr.c() {
            // This is usually I32, however it can be different on some platforms,
            // notably hexagon and arm-none/thumb-none
            interner.data_layout().c_enum_min_size
        } else {
            // repr(Rust) enums try to be as small as possible
            Integer::I8
        };

        // If there are no negative values, we can use the unsigned fit.
        if min >= 0 {
            (std::cmp::max(unsigned_fit, at_least), false)
        } else {
            (std::cmp::max(signed_fit, at_least), true)
        }
    }
}

pub trait FloatExt {
    fn to_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db>;
    fn from_float_ty(fty: FloatTy) -> Self;
}

impl FloatExt for Float {
    #[inline]
    fn to_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db> {
        use Float::*;
        match *self {
            F16 => Ty::new(interner, TyKind::Float(FloatTy::F16)),
            F32 => Ty::new(interner, TyKind::Float(FloatTy::F32)),
            F64 => Ty::new(interner, TyKind::Float(FloatTy::F64)),
            F128 => Ty::new(interner, TyKind::Float(FloatTy::F128)),
        }
    }

    fn from_float_ty(fty: FloatTy) -> Self {
        use Float::*;
        match fty {
            FloatTy::F16 => F16,
            FloatTy::F32 => F32,
            FloatTy::F64 => F64,
            FloatTy::F128 => F128,
        }
    }
}

pub trait PrimitiveExt {
    fn to_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db>;
    fn to_int_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db>;
}

impl PrimitiveExt for Primitive {
    #[inline]
    fn to_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db> {
        match *self {
            Primitive::Int(i, signed) => i.to_ty(interner, signed),
            Primitive::Float(f) => f.to_ty(interner),
            Primitive::Pointer(_) => Ty::new(
                interner,
                TyKind::RawPtr(
                    Ty::new(interner, TyKind::Tuple(Default::default())),
                    rustc_ast_ir::Mutability::Mut,
                ),
            ),
        }
    }

    /// Return an *integer* type matching this primitive.
    /// Useful in particular when dealing with enum discriminants.
    #[inline]
    fn to_int_ty<'db>(&self, interner: DbInterner<'db>) -> Ty<'db> {
        match *self {
            Primitive::Int(i, signed) => i.to_ty(interner, signed),
            Primitive::Pointer(_) => {
                let signed = false;
                interner.data_layout().ptr_sized_integer().to_ty(interner, signed)
            }
            Primitive::Float(_) => panic!("floats do not have an int type"),
        }
    }
}

impl<'db> HasDataLayout for DbInterner<'db> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        unimplemented!()
    }
}

pub trait CoroutineArgsExt<'db> {
    fn discr_ty(&self, interner: DbInterner<'db>) -> Ty<'db>;
}

impl<'db> CoroutineArgsExt<'db> for CoroutineArgs<DbInterner<'db>> {
    /// The type of the state discriminant used in the coroutine type.
    #[inline]
    fn discr_ty(&self, interner: DbInterner<'db>) -> Ty<'db> {
        Ty::new(interner, TyKind::Uint(UintTy::U32))
    }
}

/// Finds the max universe present
pub struct MaxUniverse {
    max_universe: UniverseIndex,
}

impl Default for MaxUniverse {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxUniverse {
    pub fn new() -> Self {
        MaxUniverse { max_universe: UniverseIndex::ROOT }
    }

    pub fn max_universe(self) -> UniverseIndex {
        self.max_universe
    }
}

impl<'db> TypeVisitor<DbInterner<'db>> for MaxUniverse {
    type Result = ();

    fn visit_ty(&mut self, t: Ty<'db>) {
        if let TyKind::Placeholder(placeholder) = t.kind() {
            self.max_universe = UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: Const<'db>) {
        if let ConstKind::Placeholder(placeholder) = c.kind() {
            self.max_universe = UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: Region<'db>) {
        if let RegionKind::RePlaceholder(placeholder) = r.kind() {
            self.max_universe = UniverseIndex::from_u32(
                self.max_universe.as_u32().max(placeholder.universe.as_u32()),
            );
        }
    }
}

pub struct BottomUpFolder<'db, F, G, H>
where
    F: FnMut(Ty<'db>) -> Ty<'db>,
    G: FnMut(Region<'db>) -> Region<'db>,
    H: FnMut(Const<'db>) -> Const<'db>,
{
    pub interner: DbInterner<'db>,
    pub ty_op: F,
    pub lt_op: G,
    pub ct_op: H,
}

impl<'db, F, G, H> TypeFolder<DbInterner<'db>> for BottomUpFolder<'db, F, G, H>
where
    F: FnMut(Ty<'db>) -> Ty<'db>,
    G: FnMut(Region<'db>) -> Region<'db>,
    H: FnMut(Const<'db>) -> Const<'db>,
{
    fn cx(&self) -> DbInterner<'db> {
        self.interner
    }

    fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        let t = ty.super_fold_with(self);
        (self.ty_op)(t)
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        // This one is a little different, because `super_fold_with` is not
        // implemented on non-recursive `Region`.
        (self.lt_op)(r)
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        let ct = ct.super_fold_with(self);
        (self.ct_op)(ct)
    }
}

// FIXME(next-trait-solver): uplift
pub fn sizedness_constraint_for_ty<'db>(
    interner: DbInterner<'db>,
    sizedness: SizedTraitKind,
    ty: Ty<'db>,
) -> Option<Ty<'db>> {
    use rustc_type_ir::TyKind::*;

    match ty.kind() {
        // these are always sized
        Bool | Char | Int(..) | Uint(..) | Float(..) | RawPtr(..) | Ref(..) | FnDef(..)
        | FnPtr(..) | Array(..) | Closure(..) | CoroutineClosure(..) | Coroutine(..)
        | CoroutineWitness(..) | Never => None,

        // these are never sized
        Str | Slice(..) | Dynamic(_, _) => match sizedness {
            // Never `Sized`
            SizedTraitKind::Sized => Some(ty),
            // Always `MetaSized`
            SizedTraitKind::MetaSized => None,
        },

        // Maybe `Sized` or `MetaSized`
        Param(..) | Alias(..) | Error(_) => Some(ty),

        // We cannot instantiate the binder, so just return the *original* type back,
        // but only if the inner type has a sized constraint. Thus we skip the binder,
        // but don't actually use the result from `sized_constraint_for_ty`.
        UnsafeBinder(inner_ty) => {
            sizedness_constraint_for_ty(interner, sizedness, inner_ty.skip_binder()).map(|_| ty)
        }

        // Never `MetaSized` or `Sized`
        Foreign(..) => Some(ty),

        // Recursive cases
        Pat(ty, _) => sizedness_constraint_for_ty(interner, sizedness, ty),

        Tuple(tys) => tys
            .into_iter()
            .next_back()
            .and_then(|ty| sizedness_constraint_for_ty(interner, sizedness, ty)),

        Adt(adt, args) => adt.struct_tail_ty(interner).and_then(|tail_ty| {
            let tail_ty = tail_ty.instantiate(interner, args);
            sizedness_constraint_for_ty(interner, sizedness, tail_ty)
        }),

        Placeholder(..) | Bound(..) | Infer(..) => {
            panic!("unexpected type `{ty:?}` in sizedness_constraint_for_ty")
        }
    }
}

pub fn apply_args_to_binder<'db, T: TypeFoldable<DbInterner<'db>>>(
    b: Binder<'db, T>,
    args: GenericArgs<'db>,
    interner: DbInterner<'db>,
) -> T {
    let types = &mut |ty: BoundTy| args.as_slice()[ty.var.index()].expect_ty();
    let regions = &mut |region: BoundRegion| args.as_slice()[region.var.index()].expect_region();
    let consts = &mut |const_: BoundConst| args.as_slice()[const_.var.index()].expect_const();
    let mut instantiate = BoundVarReplacer::new(interner, FnMutDelegate { types, regions, consts });
    b.skip_binder().fold_with(&mut instantiate)
}

pub fn explicit_item_bounds<'db>(
    interner: DbInterner<'db>,
    def_id: SolverDefId,
) -> EarlyBinder<'db, impl DoubleEndedIterator<Item = Clause<'db>> + ExactSizeIterator> {
    let db = interner.db();
    let clauses = match def_id {
        SolverDefId::TypeAliasId(type_alias) => crate::lower::type_alias_bounds(db, type_alias),
        SolverDefId::InternedOpaqueTyId(id) => id.predicates(db),
        _ => panic!("Unexpected GenericDefId"),
    };
    clauses.map_bound(|clauses| clauses.iter().copied())
}

pub struct ContainsTypeErrors;

impl<'db> TypeVisitor<DbInterner<'db>> for ContainsTypeErrors {
    type Result = ControlFlow<()>;

    fn visit_ty(&mut self, t: Ty<'db>) -> Self::Result {
        match t.kind() {
            rustc_type_ir::TyKind::Error(_) => ControlFlow::Break(()),
            _ => t.super_visit_with(self),
        }
    }
}

/// The inverse of [`BoundVarReplacer`]: replaces placeholders with the bound vars from which they came.
pub struct PlaceholderReplacer<'a, 'db> {
    infcx: &'a InferCtxt<'db>,
    mapped_regions: FxIndexMap<PlaceholderRegion, BoundRegion>,
    mapped_types: FxIndexMap<Placeholder<BoundTy>, BoundTy>,
    mapped_consts: FxIndexMap<PlaceholderConst, BoundConst>,
    universe_indices: &'a [Option<UniverseIndex>],
    current_index: DebruijnIndex,
}

impl<'a, 'db> PlaceholderReplacer<'a, 'db> {
    pub fn replace_placeholders<T: TypeFoldable<DbInterner<'db>>>(
        infcx: &'a InferCtxt<'db>,
        mapped_regions: FxIndexMap<PlaceholderRegion, BoundRegion>,
        mapped_types: FxIndexMap<Placeholder<BoundTy>, BoundTy>,
        mapped_consts: FxIndexMap<PlaceholderConst, BoundConst>,
        universe_indices: &'a [Option<UniverseIndex>],
        value: T,
    ) -> T {
        let mut replacer = PlaceholderReplacer {
            infcx,
            mapped_regions,
            mapped_types,
            mapped_consts,
            universe_indices,
            current_index: INNERMOST,
        };
        value.fold_with(&mut replacer)
    }
}

impl<'db> TypeFolder<DbInterner<'db>> for PlaceholderReplacer<'_, 'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.infcx.interner
    }

    fn fold_binder<T: TypeFoldable<DbInterner<'db>>>(
        &mut self,
        t: Binder<'db, T>,
    ) -> Binder<'db, T> {
        if !t.has_placeholders() && !t.has_infer() {
            return t;
        }
        self.current_index.shift_in(1);
        let t = t.super_fold_with(self);
        self.current_index.shift_out(1);
        t
    }

    fn fold_region(&mut self, r0: Region<'db>) -> Region<'db> {
        let r1 = match r0.kind() {
            RegionKind::ReVar(vid) => self
                .infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(self.infcx.interner, vid),
            _ => r0,
        };

        let r2 = match r1.kind() {
            RegionKind::RePlaceholder(p) => {
                let replace_var = self.mapped_regions.get(&p);
                match replace_var {
                    Some(replace_var) => {
                        let index = self
                            .universe_indices
                            .iter()
                            .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                            .unwrap_or_else(|| panic!("Unexpected placeholder universe."));
                        let db = DebruijnIndex::from_usize(
                            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                        );
                        Region::new_bound(self.cx(), db, *replace_var)
                    }
                    None => r1,
                }
            }
            _ => r1,
        };

        tracing::debug!(?r0, ?r1, ?r2, "fold_region");

        r2
    }

    fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        let ty = self.infcx.shallow_resolve(ty);
        match ty.kind() {
            TyKind::Placeholder(p) => {
                let replace_var = self.mapped_types.get(&p);
                match replace_var {
                    Some(replace_var) => {
                        let index = self
                            .universe_indices
                            .iter()
                            .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                            .unwrap_or_else(|| panic!("Unexpected placeholder universe."));
                        let db = DebruijnIndex::from_usize(
                            self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                        );
                        Ty::new_bound(self.infcx.interner, db, *replace_var)
                    }
                    None => {
                        if ty.has_infer() {
                            ty.super_fold_with(self)
                        } else {
                            ty
                        }
                    }
                }
            }

            _ if ty.has_placeholders() || ty.has_infer() => ty.super_fold_with(self),
            _ => ty,
        }
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        let ct = self.infcx.shallow_resolve_const(ct);
        if let ConstKind::Placeholder(p) = ct.kind() {
            let replace_var = self.mapped_consts.get(&p);
            match replace_var {
                Some(replace_var) => {
                    let index = self
                        .universe_indices
                        .iter()
                        .position(|u| matches!(u, Some(pu) if *pu == p.universe))
                        .unwrap_or_else(|| panic!("Unexpected placeholder universe."));
                    let db = DebruijnIndex::from_usize(
                        self.universe_indices.len() - index + self.current_index.as_usize() - 1,
                    );
                    Const::new_bound(self.infcx.interner, db, *replace_var)
                }
                None => {
                    if ct.has_infer() {
                        ct.super_fold_with(self)
                    } else {
                        ct
                    }
                }
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}

pub fn sizedness_fast_path<'db>(
    tcx: DbInterner<'db>,
    predicate: Predicate<'db>,
    param_env: ParamEnv<'db>,
) -> bool {
    // Proving `Sized`/`MetaSized`, very often on "obviously sized" types like
    // `&T`, accounts for about 60% percentage of the predicates we have to prove. No need to
    // canonicalize and all that for such cases.
    if let PredicateKind::Clause(ClauseKind::Trait(trait_pred)) = predicate.kind().skip_binder()
        && trait_pred.polarity == PredicatePolarity::Positive
    {
        let sizedness = match tcx.as_trait_lang_item(trait_pred.def_id()) {
            Some(SolverTraitLangItem::Sized) => SizedTraitKind::Sized,
            Some(SolverTraitLangItem::MetaSized) => SizedTraitKind::MetaSized,
            _ => return false,
        };

        // FIXME(sized_hierarchy): this temporarily reverts the `sized_hierarchy` feature
        // while a proper fix for `tests/ui/sized-hierarchy/incomplete-inference-issue-143992.rs`
        // is pending a proper fix
        if matches!(sizedness, SizedTraitKind::MetaSized) {
            return true;
        }

        if trait_pred.self_ty().has_trivial_sizedness(tcx, sizedness) {
            tracing::debug!("fast path -- trivial sizedness");
            return true;
        }

        if matches!(trait_pred.self_ty().kind(), TyKind::Param(_) | TyKind::Placeholder(_)) {
            for clause in param_env.caller_bounds().iter() {
                if let ClauseKind::Trait(clause_pred) = clause.kind().skip_binder()
                    && clause_pred.polarity == PredicatePolarity::Positive
                    && clause_pred.self_ty() == trait_pred.self_ty()
                    && (clause_pred.def_id() == trait_pred.def_id()
                        || (sizedness == SizedTraitKind::MetaSized
                            && tcx.is_trait_lang_item(
                                clause_pred.def_id(),
                                SolverTraitLangItem::Sized,
                            )))
                {
                    return true;
                }
            }
        }
    }

    false
}

/// Casts a trait reference into a reference to one of its super
/// traits; returns `None` if `target_trait_def_id` is not a
/// supertrait.
pub(crate) fn upcast_choices<'db>(
    interner: DbInterner<'db>,
    source_trait_ref: PolyTraitRef<'db>,
    target_trait_def_id: TraitId,
) -> Vec<PolyTraitRef<'db>> {
    if source_trait_ref.def_id().0 == target_trait_def_id {
        return vec![source_trait_ref]; // Shortcut the most common case.
    }

    elaborate::supertraits(interner, source_trait_ref)
        .filter(|r| r.def_id().0 == target_trait_def_id)
        .collect()
}

#[inline]
pub(crate) fn clauses_as_obligations<'db>(
    clauses: impl IntoIterator<Item = Clause<'db>>,
    cause: ObligationCause,
    param_env: ParamEnv<'db>,
) -> impl Iterator<Item = PredicateObligation<'db>> {
    clauses.into_iter().map(move |clause| Obligation {
        cause: cause.clone(),
        param_env,
        predicate: clause.as_predicate(),
        recursion_depth: 0,
    })
}
