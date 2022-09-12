//! Freshening is the process of replacing unknown variables with fresh types. The idea is that
//! the type, after freshening, contains no inference variables but instead contains either a
//! value for each variable or fresh "arbitrary" types wherever a variable would have been.
//!
//! Freshening is used primarily to get a good type for inserting into a cache. The result
//! summarizes what the type inferencer knows "so far". The primary place it is used right now is
//! in the trait matching algorithm, which needs to be able to cache whether an `impl` self type
//! matches some other type X -- *without* affecting `X`. That means if that if the type `X` is in
//! fact an unbound type variable, we want the match to be regarded as ambiguous, because depending
//! on what type that type variable is ultimately assigned, the match may or may not succeed.
//!
//! To handle closures, freshened types also have to contain the signature and kind of any
//! closure in the local inference context, as otherwise the cache key might be invalidated.
//! The way this is done is somewhat hacky - the closure signature is appended to the substs,
//! as well as the closure kind "encoded" as a type. Also, special handling is needed when
//! the closure signature contains a reference to the original closure.
//!
//! Note that you should be careful not to allow the output of freshening to leak to the user in
//! error messages or in any other form. Freshening is only really useful as an internal detail.
//!
//! Because of the manipulation required to handle closures, doing arbitrary operations on
//! freshened types is not recommended. However, in addition to doing equality/hash
//! comparisons (for caching), it is possible to do a `ty::_match` operation between
//! 2 freshened types - this works even with the closure encoding.
//!
//! __An important detail concerning regions.__ The freshener also replaces *all* free regions with
//! 'erased. The reason behind this is that, in general, we do not take region relationships into
//! account when making type-overloaded decisions. This is important because of the design of the
//! region inferencer, which is not based on unification but rather on accumulating and then
//! solving a set of constraints. In contrast, the type inferencer assigns a value to each type
//! variable only once, and it does so as soon as it can, so it is reasonable to ask what the type
//! inferencer knows "so far".
use super::InferCtxt;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::infer::unify_key::ToType;
use rustc_middle::ty::fold::TypeFolder;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable, TypeVisitable};
use std::collections::hash_map::Entry;

pub struct TypeFreshener<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    ty_freshen_count: u32,
    const_freshen_count: u32,
    ty_freshen_map: FxHashMap<ty::InferTy, Ty<'tcx>>,
    const_freshen_map: FxHashMap<ty::InferConst<'tcx>, ty::Const<'tcx>>,
    keep_static: bool,
}

impl<'a, 'tcx> TypeFreshener<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>, keep_static: bool) -> TypeFreshener<'a, 'tcx> {
        TypeFreshener {
            infcx,
            ty_freshen_count: 0,
            const_freshen_count: 0,
            ty_freshen_map: Default::default(),
            const_freshen_map: Default::default(),
            keep_static,
        }
    }

    fn freshen_ty<F>(
        &mut self,
        opt_ty: Option<Ty<'tcx>>,
        key: ty::InferTy,
        freshener: F,
    ) -> Ty<'tcx>
    where
        F: FnOnce(u32) -> ty::InferTy,
    {
        if let Some(ty) = opt_ty {
            return ty.fold_with(self);
        }

        match self.ty_freshen_map.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = self.ty_freshen_count;
                self.ty_freshen_count += 1;
                let t = self.infcx.tcx.mk_ty_infer(freshener(index));
                entry.insert(t);
                t
            }
        }
    }

    fn freshen_const<F>(
        &mut self,
        opt_ct: Option<ty::Const<'tcx>>,
        key: ty::InferConst<'tcx>,
        freshener: F,
        ty: Ty<'tcx>,
    ) -> ty::Const<'tcx>
    where
        F: FnOnce(u32) -> ty::InferConst<'tcx>,
    {
        if let Some(ct) = opt_ct {
            return ct.fold_with(self);
        }

        match self.const_freshen_map.entry(key) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let index = self.const_freshen_count;
                self.const_freshen_count += 1;
                let ct = self.infcx.tcx.mk_const_infer(freshener(index), ty);
                entry.insert(ct);
                ct
            }
        }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for TypeFreshener<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReLateBound(..) => {
                // leave bound regions alone
                r
            }

            ty::ReEarlyBound(..)
            | ty::ReFree(_)
            | ty::ReVar(_)
            | ty::RePlaceholder(..)
            | ty::ReErased => {
                // replace all free regions with 'erased
                self.tcx().lifetimes.re_erased
            }
            ty::ReStatic => {
                if self.keep_static {
                    r
                } else {
                    self.tcx().lifetimes.re_erased
                }
            }
        }
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_infer() && !t.has_erasable_regions() {
            return t;
        }

        let tcx = self.infcx.tcx;

        match *t.kind() {
            ty::Infer(ty::TyVar(v)) => {
                let opt_ty = self.infcx.inner.borrow_mut().type_variables().probe(v).known();
                self.freshen_ty(opt_ty, ty::TyVar(v), ty::FreshTy)
            }

            ty::Infer(ty::IntVar(v)) => self.freshen_ty(
                self.infcx
                    .inner
                    .borrow_mut()
                    .int_unification_table()
                    .probe_value(v)
                    .map(|v| v.to_type(tcx)),
                ty::IntVar(v),
                ty::FreshIntTy,
            ),

            ty::Infer(ty::FloatVar(v)) => self.freshen_ty(
                self.infcx
                    .inner
                    .borrow_mut()
                    .float_unification_table()
                    .probe_value(v)
                    .map(|v| v.to_type(tcx)),
                ty::FloatVar(v),
                ty::FreshFloatTy,
            ),

            ty::Infer(ty::FreshTy(ct) | ty::FreshIntTy(ct) | ty::FreshFloatTy(ct)) => {
                if ct >= self.ty_freshen_count {
                    bug!(
                        "Encountered a freshend type with id {} \
                          but our counter is only at {}",
                        ct,
                        self.ty_freshen_count
                    );
                }
                t
            }

            ty::Generator(..)
            | ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Adt(..)
            | ty::Str
            | ty::Error(_)
            | ty::Array(..)
            | ty::Slice(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::Dynamic(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Projection(..)
            | ty::Foreign(..)
            | ty::Param(..)
            | ty::Closure(..)
            | ty::GeneratorWitness(..)
            | ty::Opaque(..) => t.super_fold_with(self),

            ty::Placeholder(..) | ty::Bound(..) => bug!("unexpected type {:?}", t),
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(v)) => {
                let opt_ct = self
                    .infcx
                    .inner
                    .borrow_mut()
                    .const_unification_table()
                    .probe_value(v)
                    .val
                    .known();
                self.freshen_const(opt_ct, ty::InferConst::Var(v), ty::InferConst::Fresh, ct.ty())
            }
            ty::ConstKind::Infer(ty::InferConst::Fresh(i)) => {
                if i >= self.const_freshen_count {
                    bug!(
                        "Encountered a freshend const with id {} \
                            but our counter is only at {}",
                        i,
                        self.const_freshen_count,
                    );
                }
                ct
            }

            ty::ConstKind::Bound(..) | ty::ConstKind::Placeholder(_) => {
                bug!("unexpected const {:?}", ct)
            }

            ty::ConstKind::Param(_)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Unevaluated(..)
            | ty::ConstKind::Error(_) => ct.super_fold_with(self),
        }
    }
}
