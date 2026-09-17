//! Prove output dependencies using complete input type identities.
//!
//! Equal projections do not imply equal projection arguments. An output that
//! repeats a complete input type is nevertheless determined by that input.
//! This is independent of whether values of either type outlive a region.

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_type_ir::{TypeFoldable, TypeVisitableExt as _};

use crate::ty::{self, Binder, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor};

impl<'tcx> TyCtxt<'tcx> {
    /// Collects output lifetimes which still need to be constrained separately
    /// after accounting for the complete input types.
    ///
    /// For `for<'a> Fn(P<'a>) -> P<'a>`, the output is already the input type,
    /// even if the associated projection `P` erases `'a`. This does not make
    /// `'a` itself a constrained input: another occurrence such as the one in
    /// `(P<'a>, &'a ())` must still be checked.
    pub fn collect_output_late_bound_regions<T>(
        self,
        inputs: Binder<'tcx, Vec<Ty<'tcx>>>,
        output: Binder<'tcx, T>,
    ) -> FxIndexSet<ty::BoundRegionKind<'tcx>>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        // A supertrait can introduce another binder. Comparing terms across
        // different declarations requires an explicit rebasing first.
        if inputs.bound_vars() != output.bound_vars() {
            return self.collect_referenced_late_bound_regions(output);
        }

        let mut inputs_collector = InputTypesCollector {
            tcx: self,
            depth: 0,
            known: Default::default(),
            visited: Default::default(),
        };
        let mut anonymizer = AnonymizeNestedBinders { tcx: self, types: Default::default() };
        for input in inputs.skip_binder() {
            let input = input.fold_with(&mut anonymizer);
            input.visit_with(&mut inputs_collector);
            self.expand_free_alias_tys(input)
                .fold_with(&mut anonymizer)
                .visit_with(&mut inputs_collector);
        }
        let mut collector = OutputRegionsCollector {
            tcx: self,
            current_index: ty::INNERMOST,
            input_types: vec![inputs_collector.known],
            visited: Default::default(),
            regions: Default::default(),
        };
        // Keep output aliases intact: expanding a checked alias here could
        // hide a lifetime that still occurs in its well-formedness conditions.
        output.skip_binder().fold_with(&mut anonymizer).visit_with(&mut collector);
        collector.regions
    }
}

/// Inner binder names do not affect type identity. Keep the outer variables
/// intact so that matching cannot identify independent output lifetimes.
struct AnonymizeNestedBinders<'tcx> {
    tcx: TyCtxt<'tcx>,
    types: FxHashMap<Ty<'tcx>, Ty<'tcx>>,
}

impl<'tcx> ty::TypeFolder<TyCtxt<'tcx>> for AnonymizeNestedBinders<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        use ty::TypeSuperFoldable;
        if let Some(&ty) = self.types.get(&ty) {
            return ty;
        }
        let folded = ty.super_fold_with(self);
        self.types.insert(ty, folded);
        folded
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        binder: Binder<'tcx, T>,
    ) -> Binder<'tcx, T> {
        use ty::TypeSuperFoldable;
        self.tcx.anonymize_bound_vars(binder).super_fold_with(self)
    }
}

struct InputTypesCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    depth: u32,
    known: FxIndexSet<Ty<'tcx>>,
    visited: FxHashSet<(Ty<'tcx>, u32)>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for InputTypesCollector<'tcx> {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(&mut self, binder: &Binder<'tcx, T>) {
        self.depth += 1;
        binder.super_visit_with(self);
        self.depth -= 1;
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if !self.visited.insert((ty, self.depth)) {
            return;
        }
        if self.depth == 0 || !ty.has_escaping_bound_vars() {
            self.known.insert(ty);
        } else if let Ok(ty) = ty.try_fold_with(&mut LiftInputType {
            tcx: self.tcx,
            inner: ty::INNERMOST,
            amount: self.depth,
        }) {
            self.known.insert(ty);
        }
        // Structural type components retain their identity. Alias arguments
        // need not occur in the normalized type, so they cannot be recovered.
        if !matches!(ty.kind(), ty::Alias(..)) {
            ty.super_visit_with(self);
        }
    }

    fn visit_const(&mut self, _: ty::Const<'tcx>) {}
}

/// Move a component out of input binders only if it does not refer to their
/// variables. Binders contained within the component remain in scope.
struct LiftInputType<'tcx> {
    tcx: TyCtxt<'tcx>,
    inner: ty::DebruijnIndex,
    amount: u32,
}

impl LiftInputType<'_> {
    fn index(&self, index: ty::DebruijnIndex) -> Result<ty::DebruijnIndex, ()> {
        if index < self.inner {
            Ok(index)
        } else if index >= self.inner.shifted_in(self.amount) {
            Ok(index.shifted_out(self.amount))
        } else {
            Err(())
        }
    }
}

impl<'tcx> ty::FallibleTypeFolder<TyCtxt<'tcx>> for LiftInputType<'tcx> {
    type Error = ();

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn try_fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        binder: Binder<'tcx, T>,
    ) -> Result<Binder<'tcx, T>, Self::Error> {
        use ty::TypeSuperFoldable;
        self.inner.shift_in(1);
        let result = binder.try_super_fold_with(self);
        self.inner.shift_out(1);
        result
    }

    fn try_fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        use ty::TypeSuperFoldable;
        match *ty.kind() {
            ty::Bound(ty::BoundVarIndexKind::Bound(index), bound) => {
                Ok(Ty::new_bound(self.tcx, self.index(index)?, bound))
            }
            ty::Bound(..) => Err(()),
            _ => ty.try_super_fold_with(self),
        }
    }

    fn try_fold_region(
        &mut self,
        region: ty::Region<'tcx>,
    ) -> Result<ty::Region<'tcx>, Self::Error> {
        match region.kind() {
            ty::ReBound(ty::BoundVarIndexKind::Bound(index), bound) => {
                Ok(ty::Region::new_bound(self.tcx, self.index(index)?, bound))
            }
            ty::ReBound(..) => Err(()),
            _ => Ok(region),
        }
    }

    fn try_fold_const(&mut self, ct: ty::Const<'tcx>) -> Result<ty::Const<'tcx>, Self::Error> {
        use ty::TypeSuperFoldable;
        match ct.kind() {
            ty::ConstKind::Bound(ty::BoundVarIndexKind::Bound(index), bound) => {
                Ok(ty::Const::new_bound(self.tcx, self.index(index)?, bound))
            }
            ty::ConstKind::Bound(..) => Err(()),
            _ => ct.try_super_fold_with(self),
        }
    }
}

struct OutputRegionsCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    current_index: ty::DebruijnIndex,
    input_types: Vec<FxIndexSet<Ty<'tcx>>>,
    visited: FxHashSet<(Ty<'tcx>, ty::DebruijnIndex)>,
    regions: FxIndexSet<ty::BoundRegionKind<'tcx>>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for OutputRegionsCollector<'tcx> {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(&mut self, binder: &Binder<'tcx, T>) {
        self.current_index.shift_in(1);
        if self.current_index.as_usize() == self.input_types.len() {
            self.input_types.push(
                self.input_types[0]
                    .iter()
                    .map(|&input| ty::shift_vars(self.tcx, input, self.current_index.as_u32()))
                    .collect(),
            );
        }
        binder.super_visit_with(self);
        self.current_index.shift_out(1);
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if self.input_types[self.current_index.as_usize()].contains(&ty)
            || !self.visited.insert((ty, self.current_index))
        {
            return;
        }
        ty.super_visit_with(self);
    }

    fn visit_region(&mut self, region: ty::Region<'tcx>) {
        if let ty::ReBound(ty::BoundVarIndexKind::Bound(index), bound) = region.kind()
            && index == self.current_index
        {
            self.regions.insert(bound.kind);
        }
    }
}
