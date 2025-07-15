//! Freshening is the process of replacing unknown variables with fresh types. The idea is that
//! the type, after freshening, contains no inference variables but instead contains either a
//! value for each variable or fresh "arbitrary" types wherever a variable would have been.
//!
//! Freshening is used primarily to get a good type for inserting into a cache. The result
//! summarizes what the type inferencer knows "so far". The primary place it is used right now is
//! in the trait matching algorithm, which needs to be able to cache whether an `impl` self type
//! matches some other type X -- *without* affecting `X`. That means that if the type `X` is in
//! fact an unbound type variable, we want the match to be regarded as ambiguous, because depending
//! on what type that type variable is ultimately assigned, the match may or may not succeed.
//!
//! To handle closures, freshened types also have to contain the signature and kind of any
//! closure in the local inference context, as otherwise the cache key might be invalidated.
//! The way this is done is somewhat hacky - the closure signature is appended to the args,
//! as well as the closure kind "encoded" as a type. Also, special handling is needed when
//! the closure signature contains a reference to the original closure.
//!
//! Note that you should be careful not to allow the output of freshening to leak to the user in
//! error messages or in any other form. Freshening is only really useful as an internal detail.
//!
//! Because of the manipulation required to handle closures, doing arbitrary operations on
//! freshened types is not recommended. However, in addition to doing equality/hash
//! comparisons (for caching), it is possible to do a `ty::_match` operation between
//! two freshened types - this works even with the closure encoding.
//!
//! __An important detail concerning regions.__ The freshener also replaces *all* free regions with
//! 'erased. The reason behind this is that, in general, we do not take region relationships into
//! account when making type-overloaded decisions. This is important because of the design of the
//! region inferencer, which is not based on unification but rather on accumulating and then
//! solving a set of constraints. In contrast, the type inferencer assigns a value to each type
//! variable only once, and it does so as soon as it can, so it is reasonable to ask what the type
//! inferencer knows "so far".

use std::collections::hash_map::Entry;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::bug;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

use super::InferCtxt;

pub struct TypeFreshener<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    ty_freshen_count: u32,
    const_freshen_count: u32,
    ty_freshen_map: FxHashMap<ty::InferTy, Ty<'tcx>>,
    const_freshen_map: FxHashMap<ty::InferConst, ty::Const<'tcx>>,
}

impl<'a, 'tcx> TypeFreshener<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> TypeFreshener<'a, 'tcx> {
        TypeFreshener {
            infcx,
            ty_freshen_count: 0,
            const_freshen_count: 0,
            ty_freshen_map: Default::default(),
            const_freshen_map: Default::default(),
        }
    }

    fn freshen_ty<F>(&mut self, input: Result<Ty<'tcx>, ty::InferTy>, mk_fresh: F) -> Ty<'tcx>
    where
        F: FnOnce(u32) -> Ty<'tcx>,
    {
        match input {
            Ok(ty) => ty.fold_with(self),
            Err(key) => match self.ty_freshen_map.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let index = self.ty_freshen_count;
                    self.ty_freshen_count += 1;
                    let t = mk_fresh(index);
                    entry.insert(t);
                    t
                }
            },
        }
    }

    fn freshen_const<F>(
        &mut self,
        input: Result<ty::Const<'tcx>, ty::InferConst>,
        freshener: F,
    ) -> ty::Const<'tcx>
    where
        F: FnOnce(u32) -> ty::InferConst,
    {
        match input {
            Ok(ct) => ct.fold_with(self),
            Err(key) => match self.const_freshen_map.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let index = self.const_freshen_count;
                    self.const_freshen_count += 1;
                    let ct = ty::Const::new_infer(self.infcx.tcx, freshener(index));
                    entry.insert(ct);
                    ct
                }
            },
        }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for TypeFreshener<'a, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            // Leave bound regions alone, since they affect selection via the leak check.
            ty::ReBound(..) => r,
            // Leave error regions alone, since they affect selection b/c of incompleteness.
            ty::ReError(_) => r,

            ty::ReEarlyParam(..)
            | ty::ReLateParam(_)
            | ty::ReVar(_)
            | ty::RePlaceholder(..)
            | ty::ReStatic
            | ty::ReErased => self.cx().lifetimes.re_erased,
        }
    }

    #[inline]
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_infer() && !t.has_erasable_regions() {
            t
        } else {
            match *t.kind() {
                ty::Infer(v) => self.fold_infer_ty(v).unwrap_or(t),

                // This code is hot enough that a non-debug assertion here makes a noticeable
                // difference on benchmarks like `wg-grammar`.
                #[cfg(debug_assertions)]
                ty::Placeholder(..) | ty::Bound(..) => bug!("unexpected type {:?}", t),

                _ => t.super_fold_with(self),
            }
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(v)) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let input =
                    inner.const_unification_table().probe_value(v).known().ok_or_else(|| {
                        ty::InferConst::Var(inner.const_unification_table().find(v).vid)
                    });
                drop(inner);
                self.freshen_const(input, ty::InferConst::Fresh)
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
            | ty::ConstKind::Expr(..)
            | ty::ConstKind::Error(_) => ct.super_fold_with(self),
        }
    }
}

impl<'a, 'tcx> TypeFreshener<'a, 'tcx> {
    // This is separate from `fold_ty` to keep that method small and inlinable.
    #[inline(never)]
    fn fold_infer_ty(&mut self, v: ty::InferTy) -> Option<Ty<'tcx>> {
        match v {
            ty::TyVar(v) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let input = inner
                    .type_variables()
                    .probe(v)
                    .known()
                    .ok_or_else(|| ty::TyVar(inner.type_variables().root_var(v)));
                drop(inner);
                Some(self.freshen_ty(input, |n| Ty::new_fresh(self.infcx.tcx, n)))
            }

            ty::IntVar(v) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let value = inner.int_unification_table().probe_value(v);
                let input = match value {
                    ty::IntVarValue::IntType(ty) => Ok(Ty::new_int(self.infcx.tcx, ty)),
                    ty::IntVarValue::UintType(ty) => Ok(Ty::new_uint(self.infcx.tcx, ty)),
                    ty::IntVarValue::Unknown => {
                        Err(ty::IntVar(inner.int_unification_table().find(v)))
                    }
                };
                drop(inner);
                Some(self.freshen_ty(input, |n| Ty::new_fresh_int(self.infcx.tcx, n)))
            }

            ty::FloatVar(v) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let value = inner.float_unification_table().probe_value(v);
                let input = match value {
                    ty::FloatVarValue::Known(ty) => Ok(Ty::new_float(self.infcx.tcx, ty)),
                    ty::FloatVarValue::Unknown => {
                        Err(ty::FloatVar(inner.float_unification_table().find(v)))
                    }
                };
                drop(inner);
                Some(self.freshen_ty(input, |n| Ty::new_fresh_float(self.infcx.tcx, n)))
            }

            ty::FreshTy(ct) | ty::FreshIntTy(ct) | ty::FreshFloatTy(ct) => {
                if ct >= self.ty_freshen_count {
                    bug!(
                        "Encountered a freshend type with id {} \
                          but our counter is only at {}",
                        ct,
                        self.ty_freshen_count
                    );
                }
                None
            }
        }
    }
}
