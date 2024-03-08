use rustc_middle::infer::unify_key::{ConstVariableOriginKind, ConstVariableValue, ConstVidKey};
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::{self, ConstVid, FloatVid, IntVid, RegionVid, Ty, TyCtxt, TyVid};
use rustc_middle::ty::{TypeFoldable, TypeFolder, TypeSuperFoldable};

use crate::infer::type_variable::TypeVariableOrigin;
use crate::infer::InferCtxt;
use crate::infer::{ConstVariableOrigin, RegionVariableOrigin, UnificationTable};

use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::unify as ut;
use ut::UnifyKey;

use std::ops::Range;

use super::{NoSnapshotLeaks, VariableLengths};

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
    table: &mut UnificationTable<'_, 'tcx, ConstVidKey<'tcx>>,
    snapshot_var_len: usize,
) -> (Range<ConstVid>, Vec<ConstVariableOrigin>) {
    let range = vars_since_snapshot(table, snapshot_var_len);

    (
        range.start.vid..range.end.vid,
        (range.start.index()..range.end.index())
            .map(|index| match table.probe_value(ConstVid::from_u32(index)) {
                ConstVariableValue::Known { value: _ } => ConstVariableOrigin {
                    kind: ConstVariableOriginKind::MiscVariable,
                    span: rustc_span::DUMMY_SP,
                },
                ConstVariableValue::Unknown { origin, universe: _ } => origin,
            })
            .collect(),
    )
}

impl<'tcx> InferCtxt<'tcx> {
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
        E: NoSnapshotLeaks<'tcx>,
    {
        self.probe(|_| f().map(|value| FudgeInference(self.resolve_vars_if_possible(value))))
            .map(|FudgeInference(value)| value)
    }
}

#[macro_export]
macro_rules! fudge_vars_no_snapshot_leaks {
    ($tcx:lifetime, $t:ty) => {
        const _: () = {
            use rustc_middle::ty::TypeVisitableExt;
            use $crate::infer::snapshot::fudge::InferenceFudgeData;
            impl<$tcx> $crate::infer::snapshot::NoSnapshotLeaks<$tcx> for $t {
                type StartData = $crate::infer::snapshot::VariableLengths;
                type EndData = ($t, Option<InferenceFudgeData>);
                fn snapshot_start_data(infcx: &InferCtxt<$tcx>) -> Self::StartData {
                    infcx.variable_lengths()
                }
                fn end_of_snapshot(
                    infcx: &InferCtxt<$tcx>,
                    value: $t,
                    variable_lengths: Self::StartData,
                ) -> Self::EndData {
                    if value.has_infer() {
                        (value, Some(InferenceFudgeData::new(infcx, variable_lengths)))
                    } else {
                        (value, None)
                    }
                }
                fn avoid_leaks(
                    infcx: &InferCtxt<'tcx>,
                    (value, fudge_data): Self::EndData,
                ) -> Self {
                    if let Some(fudge_data) = fudge_data {
                        fudge_data.fudge_inference(infcx, value)
                    } else {
                        value
                    }
                }
            }
        };
    };
}

struct FudgeInference<T>(T);
impl<'tcx, T: TypeFoldable<TyCtxt<'tcx>>> NoSnapshotLeaks<'tcx> for FudgeInference<T> {
    type StartData = VariableLengths;
    type EndData = (T, Option<InferenceFudgeData>);
    fn snapshot_start_data(infcx: &InferCtxt<'tcx>) -> Self::StartData {
        infcx.variable_lengths()
    }
    fn end_of_snapshot(
        infcx: &InferCtxt<'tcx>,
        FudgeInference(value): FudgeInference<T>,
        variable_lengths: Self::StartData,
    ) -> Self::EndData {
        if value.has_infer() {
            (value, Some(InferenceFudgeData::new(infcx, variable_lengths)))
        } else {
            (value, None)
        }
    }
    fn avoid_leaks(infcx: &InferCtxt<'tcx>, (value, fudge_data): Self::EndData) -> Self {
        if let Some(fudge_data) = fudge_data {
            FudgeInference(fudge_data.fudge_inference(infcx, value))
        } else {
            FudgeInference(value)
        }
    }
}

pub struct InferenceFudgeData {
    type_vars: (Range<TyVid>, Vec<TypeVariableOrigin>),
    int_vars: Range<IntVid>,
    float_vars: Range<FloatVid>,
    region_vars: (Range<RegionVid>, Vec<RegionVariableOrigin>),
    const_vars: (Range<ConstVid>, Vec<ConstVariableOrigin>),
}

impl InferenceFudgeData {
    pub fn new<'tcx>(
        infcx: &InferCtxt<'tcx>,
        variable_lengths: VariableLengths,
    ) -> InferenceFudgeData {
        let mut inner = infcx.inner.borrow_mut();
        let type_vars = inner.type_variables().vars_since_snapshot(variable_lengths.type_vars);
        let int_vars =
            vars_since_snapshot(&mut inner.int_unification_table(), variable_lengths.int_vars);
        let float_vars =
            vars_since_snapshot(&mut inner.float_unification_table(), variable_lengths.float_vars);
        let region_vars =
            inner.unwrap_region_constraints().vars_since_snapshot(variable_lengths.region_vars);
        let const_vars = const_vars_since_snapshot(
            &mut inner.const_unification_table(),
            variable_lengths.const_vars,
        );

        InferenceFudgeData { type_vars, int_vars, float_vars, region_vars, const_vars }
    }

    pub fn fudge_inference<'tcx, T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        infcx: &InferCtxt<'tcx>,
        value: T,
    ) -> T {
        if self.type_vars.0.is_empty()
            && self.int_vars.is_empty()
            && self.float_vars.is_empty()
            && self.region_vars.0.is_empty()
            && self.const_vars.0.is_empty()
        {
            value
        } else {
            value.fold_with(&mut InferenceFudger { infcx, data: self })
        }
    }
}

struct InferenceFudger<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    data: InferenceFudgeData,
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for InferenceFudger<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        match *ty.kind() {
            ty::Infer(ty::InferTy::TyVar(vid)) => {
                if self.data.type_vars.0.contains(&vid) {
                    // This variable was created during the fudging.
                    // Recreate it with a fresh variable here.
                    let idx = vid.as_usize() - self.data.type_vars.0.start.as_usize();
                    let origin = self.data.type_vars.1[idx];
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
                if self.data.int_vars.contains(&vid) {
                    self.infcx.next_int_var()
                } else {
                    ty
                }
            }
            ty::Infer(ty::InferTy::FloatVar(vid)) => {
                if self.data.float_vars.contains(&vid) {
                    self.infcx.next_float_var()
                } else {
                    ty
                }
            }
            _ => ty.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReVar(vid) = *r
            && self.data.region_vars.0.contains(&vid)
        {
            let idx = vid.index() - self.data.region_vars.0.start.index();
            let origin = self.data.region_vars.1[idx];
            return self.infcx.next_region_var(origin);
        }
        r
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() {
            if self.data.const_vars.0.contains(&vid) {
                // This variable was created during the fudging.
                // Recreate it with a fresh variable here.
                let idx = vid.index() - self.data.const_vars.0.start.index();
                let origin = self.data.const_vars.1[idx];
                self.infcx.next_const_var(ct.ty(), origin)
            } else {
                ct
            }
        } else {
            ct.super_fold_with(self)
        }
    }
}
