use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::ty::{Ty, TyCtxt};
use rustc::ty::error::TypeError;
use rustc::ty::relate::{Relate, RelateResult, TypeRelation};

use std::collections::{HashSet, VecDeque};

/// A relation searching for items appearing at the same spot in a type.
///
/// Keeps track of item pairs found that way that correspond to item matchings not yet known.
/// This allows to match up some items that aren't exported, and which possibly even differ in
/// their names across versions.
pub struct Mismatch<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    /// The type context used.
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    /// The queue to append found item pairings.
    item_queue: VecDeque<(DefId, DefId)>,
    /// All visited items.
    visited: HashSet<(DefId, DefId)>,
}

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> Mismatch<'a, 'gcx, 'tcx> {
    /// Construct a new mismtach type relation.
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>, item_queue: VecDeque<(DefId, DefId)>)
        -> Mismatch<'a, 'gcx, 'tcx>
    {
        Mismatch {
            tcx: tcx,
            item_queue: item_queue,
            visited: Default::default(),
        }
    }

    /// Process the next pair of `DefId`s in the queue and return them.
    pub fn process_next(&mut self) -> Option<(DefId, DefId)> {
        let did_pair = self.item_queue.pop_front();

        if let Some((old_did, new_did)) = did_pair {
            let old_ty = self.tcx.type_of(old_did);
            let new_ty = self.tcx.type_of(new_did);
            let _ = self.tys(old_ty, new_ty);
        }

        did_pair
    }
}

impl<'a, 'gcx, 'tcx> TypeRelation<'a, 'gcx, 'tcx> for Mismatch<'a, 'gcx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.tcx
    }

    fn tag(&self) -> &'static str {
        "Mismatch"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(&mut self,
                                             _: ty::Variance,
                                             a: &T,
                                             b: &T)
                                             -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    fn relate<T: Relate<'tcx>>(&mut self, a: &T, b: &T) -> RelateResult<'tcx, T> {
        Relate::relate(self, a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        use rustc::ty::TypeVariants::*;

        let matching = match (&a.sty, &b.sty) {
            (&TyAdt(a_adt, _), &TyAdt(b_adt, _)) => Some((a_adt.did, b_adt.did)),
            (&TyFnDef(a_did, _, _), &TyFnDef(b_did, _, _)) |
            (&TyClosure(a_did, _), &TyClosure(b_did, _)) |
            (&TyAnon(a_did, _), &TyAnon(b_did, _)) => Some((a_did, b_did)),
            (&TyProjection(a_proj), &TyProjection(b_proj)) =>
                Some((a_proj.trait_ref.def_id, b_proj.trait_ref.def_id)),
            _ => {
                None
            },
        };

        if let Some(dids) = matching {
            if !self.visited.contains(&dids) {
                self.visited.insert(dids);
                self.item_queue.push_back(dids);
            }
        }

        relate_tys_mismatch(self, a, b)
    }

    fn regions(&mut self, a: ty::Region<'tcx>, _: ty::Region<'tcx>)
        -> RelateResult<'tcx, ty::Region<'tcx>>
    {
        // TODO
        Ok(a)
    }

    fn binders<T: Relate<'tcx>>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
        -> RelateResult<'tcx, ty::Binder<T>>
    {
        Ok(ty::Binder(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}

/// Relate two items after possible matches have been recorded.
///
/// This assumes the usage of the `Mismatch` type relation. The parametrization is used for
/// implementation-internal reasons.
fn relate_tys_mismatch<'a, 'gcx, 'tcx, R>(relation: &mut R, a: Ty<'tcx>, b: Ty<'tcx>)
    -> RelateResult<'tcx, Ty<'tcx>>
    where 'gcx: 'a + 'tcx, 'tcx: 'a, R: TypeRelation<'a, 'gcx, 'tcx>
{
    use rustc::ty::TypeVariants::*;

    let tcx = relation.tcx();
    let a_sty = &a.sty;
    let b_sty = &b.sty;
    match (a_sty, b_sty) {
        (&TyInfer(_), _) | (_, &TyInfer(_)) => {
            // As the original function this is ripped off of, we don't handle these cases.
            panic!("var types encountered in relate_tys_mismatch")
        },
        (&TyError, _) | (_, &TyError) => {
            Ok(tcx.types.err)
        },
        (&TyNever, _) |
        (&TyChar, _) |
        (&TyBool, _) |
        (&TyInt(_), _) |
        (&TyUint(_), _) |
        (&TyFloat(_), _) |
        (&TyStr, _) |
        (&TyParam(_), _) |
        (&TyClosure(_, _), _) if a == b => {
            Ok(a)
        },
        (&TyAdt(a_def, a_substs), &TyAdt(_, b_substs)) => {
            // TODO: possibly do something here
            let substs = relation.relate_item_substs(a_def.did, a_substs, b_substs)?;
            Ok(tcx.mk_adt(a_def, substs))
        },
        /* (&TyDynamic(_, _), &TyDynamic(_, _)) => {
            // TODO: decide whether this is needed
            let region_bound = relation.with_cause(Cause::ExistentialRegionBound,
                                                   |relation| {
                                                       relation.relate(a_region, b_region)
                                                   })?;
            Ok(tcx.mk_dynamic(relation.relate(a_obj, b_obj)?, region_bound))
            Err(TypeError::Mismatch)
        },*/
        (&TyRawPtr(ref a_mt), &TyRawPtr(ref b_mt)) => {
            let mt = relation.relate(a_mt, b_mt)?;
            Ok(tcx.mk_ptr(mt))
        },
        (&TyRef(a_r, a_mt), &TyRef(b_r, b_mt)) => {
            let r = relation.relate(&a_r, &b_r)?;
            let mt = relation.relate(&a_mt, &b_mt)?;
            Ok(tcx.mk_ref(r, mt))
        },
        (&TyArray(a_t, sz_a), &TyArray(b_t, sz_b)) => {
            let t = relation.relate(&a_t, &b_t)?;
            if sz_a == sz_b {
                Ok(tcx.mk_array(t, sz_a))
            } else {
                Err(TypeError::Mismatch)
            }
        },
        (&TySlice(a_t), &TySlice(b_t)) => {
            let t = relation.relate(&a_t, &b_t)?;
            Ok(tcx.mk_slice(t))
        },
        (&TyTuple(as_, a_defaulted), &TyTuple(bs, b_defaulted)) => {
            let rs = as_.iter().zip(bs).map(|(a, b)| relation.relate(a, b));
            if as_.len() == bs.len() {
                let defaulted = a_defaulted || b_defaulted;
                tcx.mk_tup(rs, defaulted)
            } else {
                Err(TypeError::Mismatch)
            }
        },
        (&TyFnDef(a_def_id, a_substs, a_fty), &TyFnDef(_, b_substs, b_fty)) => {
            let substs = ty::relate::relate_substs(relation, None, a_substs, b_substs)?;
            let fty = relation.relate(&a_fty, &b_fty)?;
            Ok(tcx.mk_fn_def(a_def_id, substs, fty))
        },
        (&TyFnPtr(a_fty), &TyFnPtr(b_fty)) => {
            let fty = relation.relate(&a_fty, &b_fty)?;
            Ok(tcx.mk_fn_ptr(fty))
        },
        (&TyProjection(ref a_data), &TyProjection(ref b_data)) => {
            let projection_ty = relation.relate(a_data, b_data)?;
            Ok(tcx.mk_projection(projection_ty.trait_ref, projection_ty.item_name(tcx)))
        },
        (&TyAnon(a_def_id, a_substs), &TyAnon(_, b_substs)) => {
            let substs = ty::relate::relate_substs(relation, None, a_substs, b_substs)?;
            Ok(tcx.mk_anon(a_def_id, substs))
        },
        _ => {
            Err(TypeError::Mismatch)
        }
    }
}
