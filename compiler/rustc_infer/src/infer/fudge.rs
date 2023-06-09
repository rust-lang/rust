use rustc_middle::ty::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::{self, ConstVid, FloatVid, IntVid, RegionVid, Ty, TyCtxt, TyVid};

use super::type_variable::TypeVariableOrigin;
use super::InferCtxt;
use super::{ConstVariableOrigin, RegionVariableOrigin, UnificationTable};

use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;
use ut::UnifyKey;

use std::ops::Range;

fn vars_since_snapshot<'tcx, T>(
    table: &mut UnificationTable<'_, 'tcx, T>,
    snapshot_var_len: usize,
) -> Range<T>
where
    T: UnifyKey,
    super::UndoLog<'tcx>: From<sv::UndoLog<ut::Delegate<T>>>,
{
    T::from_index(snapshot_var_len as u32)..T::from_index(table.len() as u32)
}

fn const_vars_since_snapshot<'tcx>(
    table: &mut UnificationTable<'_, 'tcx, ConstVid<'tcx>>,
    snapshot_var_len: usize,
) -> (Range<ConstVid<'tcx>>, Vec<ConstVariableOrigin>) {
    let range = vars_since_snapshot(table, snapshot_var_len);
    (
        range.start..range.end,
        (range.start.index..range.end.index)
            .map(|index| table.probe_value(ConstVid::from_index(index)).origin)
            .collect(),
    )
}

struct VariableLengths {
    type_var_len: usize,
    const_var_len: usize,
    int_var_len: usize,
    float_var_len: usize,
    region_constraints_len: usize,
}

impl<'tcx> InferCtxt<'tcx> {
    fn variable_lengths(&self) -> VariableLengths {
        let mut inner = self.inner.borrow_mut();
        VariableLengths {
            type_var_len: inner.type_variables().num_vars(),
            const_var_len: inner.const_unification_table().len(),
            int_var_len: inner.int_unification_table().len(),
            float_var_len: inner.float_unification_table().len(),
            region_constraints_len: inner.unwrap_region_constraints().num_region_vars(),
        }
    }

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
    #[instrument(skip(self, f), level = "debug")]
    pub fn fudge_inference_if_ok<T, E, F>(&self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let variable_lengths = self.variable_lengths();
        let (mut fudger, value) = self.probe(|_| {
            match f() {
                Ok(value) => {
                    let value = self.resolve_vars_if_possible(value);

                    // At this point, `value` could in principle refer
                    // to inference variables that have been created during
                    // the snapshot. Once we exit `probe()`, those are
                    // going to be popped, so we will have to
                    // eliminate any references to them.

                    let mut inner = self.inner.borrow_mut();
                    let type_vars =
                        inner.type_variables().vars_since_snapshot(variable_lengths.type_var_len);
                    let int_vars = vars_since_snapshot(
                        &mut inner.int_unification_table(),
                        variable_lengths.int_var_len,
                    );
                    let float_vars = vars_since_snapshot(
                        &mut inner.float_unification_table(),
                        variable_lengths.float_var_len,
                    );
                    let region_vars = inner
                        .unwrap_region_constraints()
                        .vars_since_snapshot(variable_lengths.region_constraints_len);
                    let const_vars = const_vars_since_snapshot(
                        &mut inner.const_unification_table(),
                        variable_lengths.const_var_len,
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
        if fudger.type_vars.0.is_empty()
            && fudger.int_vars.is_empty()
            && fudger.float_vars.is_empty()
            && fudger.region_vars.0.is_empty()
            && fudger.const_vars.0.is_empty()
        {
            Ok(value)
        } else {
            Ok(value.fold_with(&mut fudger))
        }
    }
}

pub struct InferenceFudger<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    type_vars: (Range<TyVid>, Vec<TypeVariableOrigin>),
    int_vars: Range<IntVid>,
    float_vars: Range<FloatVid>,
    region_vars: (Range<RegionVid>, Vec<RegionVariableOrigin>),
    const_vars: (Range<ConstVid<'tcx>>, Vec<ConstVariableOrigin>),
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for InferenceFudger<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Infer(ty::InferTy::TyVar(vid)) => {
                if self.type_vars.0.contains(&vid) {
                    // This variable was created during the fudging.
                    // Recreate it with a fresh variable here.
                    let idx = (vid.as_usize() - self.type_vars.0.start.as_usize()) as usize;
                    let origin = self.type_vars.1[idx];
                    self.infcx.next_ty_var(origin)
                } else {
                    // This variable was created before the
                    // "fudging". Since we refresh all type
                    // variables to their binding anyhow, we know
                    // that it is unbound, so we can just return
                    // it.
                    debug_assert!(
                        self.infcx.inner.borrow_mut().type_variables().probe(vid).is_unknown()
                    );
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
        if let ty::ReVar(vid) = *r && self.region_vars.0.contains(&vid) {
            let idx = vid.index() - self.region_vars.0.start.index();
            let origin = self.region_vars.1[idx];
            return self.infcx.next_region_var(origin);
        }
        r
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
            if self.const_vars.0.contains(&vid) {
                // This variable was created during the fudging.
                // Recreate it with a fresh variable here.
                let idx = (vid.index - self.const_vars.0.start.index) as usize;
                let origin = self.const_vars.1[idx];
                self.infcx.next_const_var(ct.ty(), origin)
            } else {
                ct
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}
