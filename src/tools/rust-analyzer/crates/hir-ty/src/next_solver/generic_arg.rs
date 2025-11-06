//! Things related to generic args in the next-trait-solver.

use hir_def::{GenericDefId, GenericParamId};
use macros::{TypeFoldable, TypeVisitable};
use rustc_type_ir::{
    ClosureArgs, CollectAndApply, ConstVid, CoroutineArgs, CoroutineClosureArgs, FnSigTys,
    GenericArgKind, Interner, TermKind, TyKind, TyVid, Variance,
    inherent::{GenericArg as _, GenericsOf, IntoKind, SliceLike, Term as _, Ty as _},
    relate::{Relate, VarianceDiagInfo},
};
use smallvec::SmallVec;

use crate::next_solver::{PolyFnSig, interned_vec_db};

use super::{
    Const, DbInterner, EarlyParamRegion, ErrorGuaranteed, ParamConst, Region, SolverDefId, Ty, Tys,
    generics::Generics,
};

#[derive(Copy, Clone, PartialEq, Eq, Hash, TypeVisitable, TypeFoldable)]
pub enum GenericArg<'db> {
    Ty(Ty<'db>),
    Lifetime(Region<'db>),
    Const(Const<'db>),
}

impl<'db> std::fmt::Debug for GenericArg<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ty(t) => std::fmt::Debug::fmt(t, f),
            Self::Lifetime(r) => std::fmt::Debug::fmt(r, f),
            Self::Const(c) => std::fmt::Debug::fmt(c, f),
        }
    }
}

impl<'db> GenericArg<'db> {
    pub fn ty(self) -> Option<Ty<'db>> {
        match self.kind() {
            GenericArgKind::Type(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn expect_ty(self) -> Ty<'db> {
        match self.kind() {
            GenericArgKind::Type(ty) => ty,
            _ => panic!("Expected ty, got {self:?}"),
        }
    }

    pub fn konst(self) -> Option<Const<'db>> {
        match self.kind() {
            GenericArgKind::Const(konst) => Some(konst),
            _ => None,
        }
    }

    pub fn region(self) -> Option<Region<'db>> {
        match self.kind() {
            GenericArgKind::Lifetime(r) => Some(r),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn expect_region(self) -> Region<'db> {
        match self {
            GenericArg::Lifetime(region) => region,
            _ => panic!("expected a region, got {self:?}"),
        }
    }

    pub fn error_from_id(interner: DbInterner<'db>, id: GenericParamId) -> GenericArg<'db> {
        match id {
            GenericParamId::TypeParamId(_) => Ty::new_error(interner, ErrorGuaranteed).into(),
            GenericParamId::ConstParamId(_) => Const::error(interner).into(),
            GenericParamId::LifetimeParamId(_) => Region::error(interner).into(),
        }
    }
}

impl<'db> From<Term<'db>> for GenericArg<'db> {
    fn from(value: Term<'db>) -> Self {
        match value {
            Term::Ty(ty) => GenericArg::Ty(ty),
            Term::Const(c) => GenericArg::Const(c),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TypeVisitable, TypeFoldable)]
pub enum Term<'db> {
    Ty(Ty<'db>),
    Const(Const<'db>),
}

impl<'db> std::fmt::Debug for Term<'db> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ty(t) => std::fmt::Debug::fmt(t, f),
            Self::Const(c) => std::fmt::Debug::fmt(c, f),
        }
    }
}

impl<'db> Term<'db> {
    pub fn expect_type(&self) -> Ty<'db> {
        self.as_type().expect("expected a type, but found a const")
    }

    pub fn is_trivially_wf(&self, tcx: DbInterner<'db>) -> bool {
        match self.kind() {
            TermKind::Ty(ty) => ty.is_trivially_wf(tcx),
            TermKind::Const(ct) => ct.is_trivially_wf(),
        }
    }
}

impl<'db> From<Ty<'db>> for GenericArg<'db> {
    fn from(value: Ty<'db>) -> Self {
        Self::Ty(value)
    }
}

impl<'db> From<Region<'db>> for GenericArg<'db> {
    fn from(value: Region<'db>) -> Self {
        Self::Lifetime(value)
    }
}

impl<'db> From<Const<'db>> for GenericArg<'db> {
    fn from(value: Const<'db>) -> Self {
        Self::Const(value)
    }
}

impl<'db> IntoKind for GenericArg<'db> {
    type Kind = GenericArgKind<DbInterner<'db>>;

    fn kind(self) -> Self::Kind {
        match self {
            GenericArg::Ty(ty) => GenericArgKind::Type(ty),
            GenericArg::Lifetime(region) => GenericArgKind::Lifetime(region),
            GenericArg::Const(c) => GenericArgKind::Const(c),
        }
    }
}

impl<'db> Relate<DbInterner<'db>> for GenericArg<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        match (a.kind(), b.kind()) {
            (GenericArgKind::Lifetime(a_lt), GenericArgKind::Lifetime(b_lt)) => {
                Ok(relation.relate(a_lt, b_lt)?.into())
            }
            (GenericArgKind::Type(a_ty), GenericArgKind::Type(b_ty)) => {
                Ok(relation.relate(a_ty, b_ty)?.into())
            }
            (GenericArgKind::Const(a_ct), GenericArgKind::Const(b_ct)) => {
                Ok(relation.relate(a_ct, b_ct)?.into())
            }
            (GenericArgKind::Lifetime(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (GenericArgKind::Type(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (GenericArgKind::Const(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
        }
    }
}

interned_vec_db!(GenericArgs, GenericArg);

impl<'db> rustc_type_ir::inherent::GenericArg<DbInterner<'db>> for GenericArg<'db> {}

impl<'db> GenericArgs<'db> {
    /// Creates an `GenericArgs` for generic parameter definitions,
    /// by calling closures to obtain each kind.
    /// The closures get to observe the `GenericArgs` as they're
    /// being built, which can be used to correctly
    /// replace defaults of generic parameters.
    pub fn for_item<F>(
        interner: DbInterner<'db>,
        def_id: SolverDefId,
        mut mk_kind: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let defs = interner.generics_of(def_id);
        let count = defs.count();
        let mut args = SmallVec::with_capacity(count);
        Self::fill_item(&mut args, interner, defs, &mut mk_kind);
        interner.mk_args(&args)
    }

    /// Creates an all-error `GenericArgs`.
    pub fn error_for_item(interner: DbInterner<'db>, def_id: SolverDefId) -> GenericArgs<'db> {
        GenericArgs::for_item(interner, def_id, |_, id, _| GenericArg::error_from_id(interner, id))
    }

    /// Like `for_item`, but prefers the default of a parameter if it has any.
    pub fn for_item_with_defaults<F>(
        interner: DbInterner<'db>,
        def_id: GenericDefId,
        mut fallback: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let defaults = interner.db.generic_defaults(def_id);
        Self::for_item(interner, def_id.into(), |idx, id, prev| match defaults.get(idx as usize) {
            Some(default) => default.instantiate(interner, prev),
            None => fallback(idx, id, prev),
        })
    }

    /// Like `for_item()`, but calls first uses the args from `first`.
    pub fn fill_rest<F>(
        interner: DbInterner<'db>,
        def_id: SolverDefId,
        first: impl IntoIterator<Item = GenericArg<'db>>,
        mut fallback: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let mut iter = first.into_iter();
        Self::for_item(interner, def_id, |idx, id, prev| {
            iter.next().unwrap_or_else(|| fallback(idx, id, prev))
        })
    }

    /// Appends default param values to `first` if needed. Params without default will call `fallback()`.
    pub fn fill_with_defaults<F>(
        interner: DbInterner<'db>,
        def_id: GenericDefId,
        first: impl IntoIterator<Item = GenericArg<'db>>,
        mut fallback: F,
    ) -> GenericArgs<'db>
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        let defaults = interner.db.generic_defaults(def_id);
        Self::fill_rest(interner, def_id.into(), first, |idx, id, prev| {
            defaults
                .get(idx as usize)
                .map(|default| default.instantiate(interner, prev))
                .unwrap_or_else(|| fallback(idx, id, prev))
        })
    }

    fn fill_item<F>(
        args: &mut SmallVec<[GenericArg<'db>; 8]>,
        interner: DbInterner<'_>,
        defs: Generics,
        mk_kind: &mut F,
    ) where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        if let Some(def_id) = defs.parent {
            let parent_defs = interner.generics_of(def_id.into());
            Self::fill_item(args, interner, parent_defs, mk_kind);
        }
        Self::fill_single(args, &defs, mk_kind);
    }

    fn fill_single<F>(args: &mut SmallVec<[GenericArg<'db>; 8]>, defs: &Generics, mk_kind: &mut F)
    where
        F: FnMut(u32, GenericParamId, &[GenericArg<'db>]) -> GenericArg<'db>,
    {
        args.reserve(defs.own_params.len());
        for param in &defs.own_params {
            let kind = mk_kind(args.len() as u32, param.id, args);
            args.push(kind);
        }
    }

    pub fn closure_sig_untupled(self) -> PolyFnSig<'db> {
        let TyKind::FnPtr(inputs_and_output, hdr) =
            self.split_closure_args_untupled().closure_sig_as_fn_ptr_ty.kind()
        else {
            unreachable!("not a function pointer")
        };
        inputs_and_output.with(hdr)
    }

    /// A "sensible" `.split_closure_args()`, where the arguments are not in a tuple.
    pub fn split_closure_args_untupled(self) -> rustc_type_ir::ClosureArgsParts<DbInterner<'db>> {
        // FIXME: should use `ClosureSubst` when possible
        match self.inner().as_slice() {
            [parent_args @ .., closure_kind_ty, sig_ty, tupled_upvars_ty] => {
                let interner = DbInterner::conjure();
                rustc_type_ir::ClosureArgsParts {
                    parent_args: GenericArgs::new_from_iter(interner, parent_args.iter().cloned()),
                    closure_sig_as_fn_ptr_ty: sig_ty.expect_ty(),
                    closure_kind_ty: closure_kind_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => {
                unreachable!("unexpected closure sig");
            }
        }
    }

    pub fn types(self) -> impl Iterator<Item = Ty<'db>> {
        self.iter().filter_map(|it| it.as_type())
    }

    pub fn consts(self) -> impl Iterator<Item = Const<'db>> {
        self.iter().filter_map(|it| it.as_const())
    }

    pub fn regions(self) -> impl Iterator<Item = Region<'db>> {
        self.iter().filter_map(|it| it.as_region())
    }
}

impl<'db> rustc_type_ir::relate::Relate<DbInterner<'db>> for GenericArgs<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        let interner = relation.cx();
        CollectAndApply::collect_and_apply(
            std::iter::zip(a.iter(), b.iter()).map(|(a, b)| {
                relation.relate_with_variance(
                    Variance::Invariant,
                    VarianceDiagInfo::default(),
                    a,
                    b,
                )
            }),
            |g| GenericArgs::new_from_iter(interner, g.iter().cloned()),
        )
    }
}

impl<'db> rustc_type_ir::inherent::GenericArgs<DbInterner<'db>> for GenericArgs<'db> {
    fn as_closure(self) -> ClosureArgs<DbInterner<'db>> {
        ClosureArgs { args: self }
    }
    fn as_coroutine(self) -> CoroutineArgs<DbInterner<'db>> {
        CoroutineArgs { args: self }
    }
    fn as_coroutine_closure(self) -> CoroutineClosureArgs<DbInterner<'db>> {
        CoroutineClosureArgs { args: self }
    }
    fn rebase_onto(
        self,
        interner: DbInterner<'db>,
        source_def_id: <DbInterner<'db> as rustc_type_ir::Interner>::DefId,
        target: <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs,
    ) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        let defs = interner.generics_of(source_def_id);
        interner.mk_args_from_iter(target.iter().chain(self.iter().skip(defs.count())))
    }

    fn identity_for_item(
        interner: DbInterner<'db>,
        def_id: <DbInterner<'db> as rustc_type_ir::Interner>::DefId,
    ) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        Self::for_item(interner, def_id, |index, kind, _| mk_param(interner, index, kind))
    }

    fn extend_with_error(
        interner: DbInterner<'db>,
        def_id: <DbInterner<'db> as rustc_type_ir::Interner>::DefId,
        original_args: &[<DbInterner<'db> as rustc_type_ir::Interner>::GenericArg],
    ) -> <DbInterner<'db> as rustc_type_ir::Interner>::GenericArgs {
        Self::for_item(interner, def_id, |index, kind, _| {
            if let Some(arg) = original_args.get(index as usize) {
                *arg
            } else {
                error_for_param_kind(kind, interner)
            }
        })
    }
    fn type_at(self, i: usize) -> <DbInterner<'db> as rustc_type_ir::Interner>::Ty {
        self.inner()
            .get(i)
            .and_then(|g| g.as_type())
            .unwrap_or_else(|| Ty::new_error(DbInterner::conjure(), ErrorGuaranteed))
    }

    fn region_at(self, i: usize) -> <DbInterner<'db> as rustc_type_ir::Interner>::Region {
        self.inner()
            .get(i)
            .and_then(|g| g.as_region())
            .unwrap_or_else(|| Region::error(DbInterner::conjure()))
    }

    fn const_at(self, i: usize) -> <DbInterner<'db> as rustc_type_ir::Interner>::Const {
        self.inner()
            .get(i)
            .and_then(|g| g.as_const())
            .unwrap_or_else(|| Const::error(DbInterner::conjure()))
    }

    fn split_closure_args(self) -> rustc_type_ir::ClosureArgsParts<DbInterner<'db>> {
        // FIXME: should use `ClosureSubst` when possible
        match self.inner().as_slice() {
            [parent_args @ .., closure_kind_ty, sig_ty, tupled_upvars_ty] => {
                let interner = DbInterner::conjure();
                // This is stupid, but the next solver expects the first input to actually be a tuple
                let sig_ty = match sig_ty.expect_ty().kind() {
                    TyKind::FnPtr(sig_tys, header) => Ty::new(
                        interner,
                        TyKind::FnPtr(
                            sig_tys.map_bound(|s| {
                                let inputs = Ty::new_tup_from_iter(interner, s.inputs().iter());
                                let output = s.output();
                                FnSigTys {
                                    inputs_and_output: Tys::new_from_iter(
                                        interner,
                                        [inputs, output],
                                    ),
                                }
                            }),
                            header,
                        ),
                    ),
                    _ => unreachable!("sig_ty should be last"),
                };
                rustc_type_ir::ClosureArgsParts {
                    parent_args: GenericArgs::new_from_iter(interner, parent_args.iter().cloned()),
                    closure_sig_as_fn_ptr_ty: sig_ty,
                    closure_kind_ty: closure_kind_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => {
                unreachable!("unexpected closure sig");
            }
        }
    }

    fn split_coroutine_closure_args(
        self,
    ) -> rustc_type_ir::CoroutineClosureArgsParts<DbInterner<'db>> {
        match self.inner().as_slice() {
            [
                parent_args @ ..,
                closure_kind_ty,
                signature_parts_ty,
                tupled_upvars_ty,
                coroutine_captures_by_ref_ty,
            ] => rustc_type_ir::CoroutineClosureArgsParts {
                parent_args: GenericArgs::new_from_iter(
                    DbInterner::conjure(),
                    parent_args.iter().cloned(),
                ),
                closure_kind_ty: closure_kind_ty.expect_ty(),
                signature_parts_ty: signature_parts_ty.expect_ty(),
                tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                coroutine_captures_by_ref_ty: coroutine_captures_by_ref_ty.expect_ty(),
            },
            _ => panic!("GenericArgs were likely not for a CoroutineClosure."),
        }
    }

    fn split_coroutine_args(self) -> rustc_type_ir::CoroutineArgsParts<DbInterner<'db>> {
        let interner = DbInterner::conjure();
        match self.inner().as_slice() {
            [parent_args @ .., kind_ty, resume_ty, yield_ty, return_ty, tupled_upvars_ty] => {
                rustc_type_ir::CoroutineArgsParts {
                    parent_args: GenericArgs::new_from_iter(interner, parent_args.iter().cloned()),
                    kind_ty: kind_ty.expect_ty(),
                    resume_ty: resume_ty.expect_ty(),
                    yield_ty: yield_ty.expect_ty(),
                    return_ty: return_ty.expect_ty(),
                    tupled_upvars_ty: tupled_upvars_ty.expect_ty(),
                }
            }
            _ => panic!("GenericArgs were likely not for a Coroutine."),
        }
    }
}

pub fn mk_param<'db>(interner: DbInterner<'db>, index: u32, id: GenericParamId) -> GenericArg<'db> {
    match id {
        GenericParamId::LifetimeParamId(id) => {
            Region::new_early_param(interner, EarlyParamRegion { index, id }).into()
        }
        GenericParamId::TypeParamId(id) => Ty::new_param(interner, id, index).into(),
        GenericParamId::ConstParamId(id) => {
            Const::new_param(interner, ParamConst { index, id }).into()
        }
    }
}

pub fn error_for_param_kind<'db>(id: GenericParamId, interner: DbInterner<'db>) -> GenericArg<'db> {
    match id {
        GenericParamId::LifetimeParamId(_) => Region::error(interner).into(),
        GenericParamId::TypeParamId(_) => Ty::new_error(interner, ErrorGuaranteed).into(),
        GenericParamId::ConstParamId(_) => Const::error(interner).into(),
    }
}

impl<'db> IntoKind for Term<'db> {
    type Kind = TermKind<DbInterner<'db>>;

    fn kind(self) -> Self::Kind {
        match self {
            Term::Ty(ty) => TermKind::Ty(ty),
            Term::Const(c) => TermKind::Const(c),
        }
    }
}

impl<'db> From<Ty<'db>> for Term<'db> {
    fn from(value: Ty<'db>) -> Self {
        Self::Ty(value)
    }
}

impl<'db> From<Const<'db>> for Term<'db> {
    fn from(value: Const<'db>) -> Self {
        Self::Const(value)
    }
}

impl<'db> Relate<DbInterner<'db>> for Term<'db> {
    fn relate<R: rustc_type_ir::relate::TypeRelation<DbInterner<'db>>>(
        relation: &mut R,
        a: Self,
        b: Self,
    ) -> rustc_type_ir::relate::RelateResult<DbInterner<'db>, Self> {
        match (a.kind(), b.kind()) {
            (TermKind::Ty(a_ty), TermKind::Ty(b_ty)) => Ok(relation.relate(a_ty, b_ty)?.into()),
            (TermKind::Const(a_ct), TermKind::Const(b_ct)) => {
                Ok(relation.relate(a_ct, b_ct)?.into())
            }
            (TermKind::Ty(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
            (TermKind::Const(unpacked), x) => {
                unreachable!("impossible case reached: can't relate: {:?} with {:?}", unpacked, x)
            }
        }
    }
}

impl<'db> rustc_type_ir::inherent::Term<DbInterner<'db>> for Term<'db> {}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum TermVid {
    Ty(TyVid),
    Const(ConstVid),
}

impl From<TyVid> for TermVid {
    fn from(value: TyVid) -> Self {
        TermVid::Ty(value)
    }
}

impl From<ConstVid> for TermVid {
    fn from(value: ConstVid) -> Self {
        TermVid::Const(value)
    }
}

impl<'db> DbInterner<'db> {
    pub(super) fn mk_args(self, args: &[GenericArg<'db>]) -> GenericArgs<'db> {
        GenericArgs::new_from_iter(self, args.iter().cloned())
    }

    pub(super) fn mk_args_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: rustc_type_ir::CollectAndApply<GenericArg<'db>, GenericArgs<'db>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_args(xs))
    }
}
