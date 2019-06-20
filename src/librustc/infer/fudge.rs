use crate::ty::{self, Ty, TyCtxt, TyVid, IntVid, FloatVid, RegionVid, ConstVid};
use crate::ty::fold::{TypeFoldable, TypeFolder};
use crate::mir::interpret::ConstValue;

use super::InferCtxt;
use super::{RegionVariableOrigin, ConstVariableOrigin};
use super::type_variable::TypeVariableOrigin;

use rustc_data_structures::unify as ut;
use ut::UnifyKey;

use std::cell::RefMut;
use std::ops::Range;

fn const_vars_since_snapshot<'tcx>(
    mut table: RefMut<'_, ut::UnificationTable<ut::InPlace<ConstVid<'tcx>>>>,
    snapshot: &ut::Snapshot<ut::InPlace<ConstVid<'tcx>>>,
) -> (Range<ConstVid<'tcx>>, Vec<ConstVariableOrigin>) {
    let range = table.vars_since_snapshot(snapshot);
    (range.start..range.end, (range.start.index..range.end.index).map(|index| {
        table.probe_value(ConstVid::from_index(index)).origin.clone()
    }).collect())
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    /// This rather funky routine is used while processing expected
    /// types. What happens here is that we want to propagate a
    /// coercion through the return type of a fn to its
    /// argument. Consider the type of `Option::Some`, which is
    /// basically `for<T> fn(T) -> Option<T>`. So if we have an
    /// expression `Some(&[1, 2, 3])`, and that has the expected type
    /// `Option<&[u32]>`, we would like to type check `&[1, 2, 3]`
    /// with the expectation of `&[u32]`. This will cause us to coerce
    /// from `&[u32; 3]` to `&[u32]` and make the users life more
    /// pleasant.
    ///
    /// The way we do this is using `fudge_inference_if_ok`. What the
    /// routine actually does is to start a snapshot and execute the
    /// closure `f`. In our example above, what this closure will do
    /// is to unify the expectation (`Option<&[u32]>`) with the actual
    /// return type (`Option<?T>`, where `?T` represents the variable
    /// instantiated for `T`). This will cause `?T` to be unified
    /// with `&?a [u32]`, where `?a` is a fresh lifetime variable. The
    /// input type (`?T`) is then returned by `f()`.
    ///
    /// At this point, `fudge_inference_if_ok` will normalize all type
    /// variables, converting `?T` to `&?a [u32]` and end the
    /// snapshot. The problem is that we can't just return this type
    /// out, because it references the region variable `?a`, and that
    /// region variable was popped when we popped the snapshot.
    ///
    /// So what we do is to keep a list (`region_vars`, in the code below)
    /// of region variables created during the snapshot (here, `?a`). We
    /// fold the return value and replace any such regions with a *new*
    /// region variable (e.g., `?b`) and return the result (`&?b [u32]`).
    /// This can then be used as the expectation for the fn argument.
    ///
    /// The important point here is that, for soundness purposes, the
    /// regions in question are not particularly important. We will
    /// use the expected types to guide coercions, but we will still
    /// type-check the resulting types from those coercions against
    /// the actual types (`?T`, `Option<?T>`) -- and remember that
    /// after the snapshot is popped, the variable `?T` is no longer
    /// unified.
    pub fn fudge_inference_if_ok<T, E, F>(
        &self,
        f: F,
    ) -> Result<T, E> where
        F: FnOnce() -> Result<T, E>,
        T: TypeFoldable<'tcx>,
    {
        debug!("fudge_inference_if_ok()");

        let (mut fudger, value) = self.probe(|snapshot| {
            match f() {
                Ok(value) => {
                    let value = self.resolve_vars_if_possible(&value);

                    // At this point, `value` could in principle refer
                    // to inference variables that have been created during
                    // the snapshot. Once we exit `probe()`, those are
                    // going to be popped, so we will have to
                    // eliminate any references to them.

                    let type_vars = self.type_variables.borrow_mut().vars_since_snapshot(
                        &snapshot.type_snapshot,
                    );
                    let int_vars = self.int_unification_table.borrow_mut().vars_since_snapshot(
                        &snapshot.int_snapshot,
                    );
                    let float_vars = self.float_unification_table.borrow_mut().vars_since_snapshot(
                        &snapshot.float_snapshot,
                    );
                    let region_vars = self.borrow_region_constraints().vars_since_snapshot(
                        &snapshot.region_constraints_snapshot,
                    );
                    let const_vars = const_vars_since_snapshot(
                        self.const_unification_table.borrow_mut(),
                        &snapshot.const_snapshot,
                    );

                    let fudger = InferenceFudger {
                        infcx: self,
                        type_vars,
                        int_vars,
                        float_vars,
                        region_vars,
                        const_vars,
                    };

                    Ok((fudger, value))
                }
                Err(e) => Err(e),
            }
        })?;

        // At this point, we need to replace any of the now-popped
        // type/region variables that appear in `value` with a fresh
        // variable of the appropriate kind. We can't do this during
        // the probe because they would just get popped then too. =)

        // Micro-optimization: if no variables have been created, then
        // `value` can't refer to any of them. =) So we can just return it.
        if fudger.type_vars.0.is_empty() &&
            fudger.int_vars.is_empty() &&
            fudger.float_vars.is_empty() &&
            fudger.region_vars.0.is_empty() &&
            fudger.const_vars.0.is_empty() {
            Ok(value)
        } else {
            Ok(value.fold_with(&mut fudger))
        }
    }
}

pub struct InferenceFudger<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    type_vars: (Range<TyVid>, Vec<TypeVariableOrigin>),
    int_vars: Range<IntVid>,
    float_vars: Range<FloatVid>,
    region_vars: (Range<RegionVid>, Vec<RegionVariableOrigin>),
    const_vars: (Range<ConstVid<'tcx>>, Vec<ConstVariableOrigin>),
}

impl<'a, 'tcx> TypeFolder<'tcx> for InferenceFudger<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match ty.sty {
            ty::Infer(ty::InferTy::TyVar(vid)) => {
                if self.type_vars.0.contains(&vid) {
                    // This variable was created during the fudging.
                    // Recreate it with a fresh variable here.
                    let idx = (vid.index - self.type_vars.0.start.index) as usize;
                    let origin = self.type_vars.1[idx];
                    self.infcx.next_ty_var(origin)
                } else {
                    // This variable was created before the
                    // "fudging". Since we refresh all type
                    // variables to their binding anyhow, we know
                    // that it is unbound, so we can just return
                    // it.
                    debug_assert!(self.infcx.type_variables.borrow_mut()
                                  .probe(vid)
                                  .is_unknown());
                    ty
                }
            }
            ty::Infer(ty::InferTy::IntVar(vid)) => {
                if self.int_vars.contains(&vid) {
                    self.infcx.next_int_var()
                } else {
                    ty
                }
            }
            ty::Infer(ty::InferTy::FloatVar(vid)) => {
                if self.float_vars.contains(&vid) {
                    self.infcx.next_float_var()
                } else {
                    ty
                }
            }
            _ => ty.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReVar(vid) = *r {
            if self.region_vars.0.contains(&vid) {
                let idx = vid.index() - self.region_vars.0.start.index();
                let origin = self.region_vars.1[idx];
                return self.infcx.next_region_var(origin);
            }
        }
        r
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if let ty::Const { val: ConstValue::Infer(ty::InferConst::Var(vid)), ty } = ct {
            if self.const_vars.0.contains(&vid) {
                // This variable was created during the fudging.
                // Recreate it with a fresh variable here.
                let idx = (vid.index - self.const_vars.0.start.index) as usize;
                let origin = self.const_vars.1[idx];
                self.infcx.next_const_var(ty, origin)
            } else {
                ct
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}
