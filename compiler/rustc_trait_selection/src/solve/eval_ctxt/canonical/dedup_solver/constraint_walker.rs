use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty;
use rustc_middle::ty::{
    Const, GenericArg, Region, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
};

pub struct DedupWalker<'me, 'tcx> {
    infcx: &'me InferCtxt<'tcx>,
    var_indexer: &'me mut DedupableIndexer<'tcx>,
    max_nameable_universe: ty::UniverseIndex,

    vars_present: Vec<usize>,
}
pub struct DedupableIndexer<'tcx> {
    vars: FxIndexSet<GenericArg<'tcx>>,
    pub unremovable_vars: FxIndexSet<usize>,
}

impl<'me, 'tcx> DedupWalker<'me, 'tcx> {
    pub fn erase_dedupables<T: TypeFoldable<TyCtxt<'tcx>>>(
        infcx: &'me InferCtxt<'tcx>,
        var_indexer: &'me mut DedupableIndexer<'tcx>,
        max_nameable_universe: ty::UniverseIndex,
        value: T,
    ) -> (T, Vec<usize>) {
        let mut dedup_walker =
            Self { infcx, var_indexer, max_nameable_universe, vars_present: Vec::new() };
        let folded = value.fold_with(&mut dedup_walker);
        (folded, dedup_walker.vars_present)
    }
}
impl<'tcx> DedupableIndexer<'tcx> {
    pub fn new() -> Self {
        Self { vars: FxIndexSet::default(), unremovable_vars: FxIndexSet::default() }
    }
    fn lookup(&mut self, var: GenericArg<'tcx>) -> usize {
        self.vars.get_index_of(&var).unwrap_or_else(|| self.vars.insert_full(var).0)
    }
    fn add_unremovable_var(&mut self, var: usize) {
        self.unremovable_vars.insert(var);
    }
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for DedupWalker<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        t.super_fold_with(self)
    }

    fn fold_region(&mut self, region: Region<'tcx>) -> Region<'tcx> {
        let universe = match *region {
            ty::ReVar(..) | ty::RePlaceholder(..) => self.infcx.universe_of_region(region),
            _ => return region,
        };
        let var_id = self.var_indexer.lookup(GenericArg::from(region));
        self.vars_present.push(var_id);
        if self.max_nameable_universe.can_name(universe) {
            self.var_indexer.add_unremovable_var(var_id);
        }
        // dummy value
        self.interner().mk_re_placeholder(ty::Placeholder {
            universe: ty::UniverseIndex::from(self.max_nameable_universe.index() + 1),
            bound: ty::BoundRegion {
                var: ty::BoundVar::from_usize(0),
                kind: ty::BoundRegionKind::BrEnv,
            },
        })
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let universe = match *ty.kind() {
            ty::Placeholder(p) => p.universe,
            ty::Infer(ty::InferTy::TyVar(vid)) => {
                if let Err(uni) = self.infcx.probe_ty_var(vid) {
                    uni
                } else {
                    return ty;
                }
            }
            _ => return ty,
        };
        let var_id = self.var_indexer.lookup(GenericArg::from(ty));
        self.vars_present.push(var_id);
        if self.max_nameable_universe.can_name(universe) {
            self.var_indexer.add_unremovable_var(var_id);
        }
        // dummy value
        self.interner().mk_ty_from_kind(ty::Placeholder(ty::Placeholder {
            universe: ty::UniverseIndex::from(self.max_nameable_universe.index() + 1),
            bound: ty::BoundTy { var: ty::BoundVar::from_usize(0), kind: ty::BoundTyKind::Anon },
        }))
    }

    fn fold_const(&mut self, ct: Const<'tcx>) -> Const<'tcx> {
        let new_ty = self.fold_ty(ct.ty());
        let universe = match ct.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                if let Err(uni) = self.infcx.probe_const_var(vid) { Some(uni) } else { None }
            }
            ty::ConstKind::Placeholder(p) => Some(p.universe),
            _ => None,
        };
        let new_const_kind = if let Some(uni) = universe {
            let var_id = self.var_indexer.lookup(GenericArg::from(ct));
            self.vars_present.push(var_id);
            if self.max_nameable_universe.can_name(uni) {
                self.var_indexer.add_unremovable_var(var_id);
            }
            // dummy value
            ty::ConstKind::Placeholder(ty::Placeholder {
                universe: ty::UniverseIndex::from(self.max_nameable_universe.index() + 1),
                bound: ty::BoundVar::from_usize(0),
            })
        } else {
            ct.kind()
        };
        self.infcx.tcx.mk_const(new_const_kind, new_ty)
    }
}
