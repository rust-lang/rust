//! The logic for the second analysis pass collecting mismatched non-public items to match them.
//!
//! Any two items' types found in the same place which are yet not matched with other items are
//! essentially just renamed instances of the same item (as long as they are both unknown to us
//! at the time of analysis). Thus, we may match them up to avoid some false positives.

use rustc::hir::def_id::{CrateNum, DefId};
use rustc::ty;
use rustc::ty::{Ty, TyCtxt};
use rustc::ty::Visibility::Public;
use rustc::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc::ty::subst::Substs;

use semcheck::mapping::IdMapping;

use std::collections::{HashMap, VecDeque};

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
    /// The id mapping to use.
    id_mapping: &'a mut IdMapping,
    /// The old crate.
    old_crate: CrateNum,
}

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> Mismatch<'a, 'gcx, 'tcx> {
    /// Construct a new mismtach type relation.
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>, old_crate: CrateNum, id_mapping: &'a mut IdMapping)
        -> Mismatch<'a, 'gcx, 'tcx>
    {
        Mismatch {
            tcx: tcx,
            item_queue: id_mapping.toplevel_queue(),
            id_mapping: id_mapping,
            old_crate: old_crate,
        }
    }

    /// Process the next pair of `DefId`s in the queue and return them.
    pub fn process(&mut self) {
        use rustc::hir::def::Def::*;

        while let Some((old_did, new_did)) = self.item_queue.pop_front() {
            match self.tcx.describe_def(old_did) {
                Some(Trait(_)) | Some(Macro(_, _)) => continue,
                _ => (),
            }

            let old_ty = self.tcx.type_of(old_did);
            let new_ty = self.tcx.type_of(new_did);
            let _ = self.relate(&old_ty, &new_ty);
        }
    }

    fn check_substs(&self, a_substs: &'tcx Substs<'tcx>, b_substs: &'tcx Substs<'tcx>)
        -> bool
    {
        for (a, b) in a_substs.iter().zip(b_substs) {
            if a.as_type().is_some() != b.as_type().is_some() {
                return false;
            }
        }

        true
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

        // TODO: maybe fetch def ids from TyClosure
        let matching = match (&a.sty, &b.sty) {
            (&TyAdt(a_def, a_substs), &TyAdt(b_def, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let _ = self.relate_item_substs(a_def.did, a_substs, b_substs)?;
                    let a_adt = self.tcx.adt_def(a_def.did);
                    let b_adt = self.tcx.adt_def(b_def.did);

                    let b_fields: HashMap<_, _> =
                        b_adt
                            .all_fields()
                            .map(|f| (f.did, f))
                            .collect();

                    for field in a_adt.all_fields().filter(|f| f.vis == Public) {
                        if self.id_mapping.contains_id(field.did) {
                            let a_field_ty = field.ty(self.tcx, a_substs);
                            let b_field_ty =
                                b_fields[&self.id_mapping.get_new_id(field.did)]
                                    .ty(self.tcx, b_substs);

                            let _ = self.relate(&a_field_ty, &b_field_ty)?;
                        }
                    }

                    Some((a_def.did, b_def.did))
                } else {
                    None
                }
            },
            (&TyArray(a_t, _), &TyArray(b_t, _)) |
            (&TySlice(a_t), &TySlice(b_t)) => {
                let _ = self.relate(&a_t, &b_t)?;
                None
            },
            (&TyRawPtr(a_mt), &TyRawPtr(b_mt)) => {
                let _ = self.relate(&a_mt, &b_mt)?;
                None
            },
            (&TyRef(a_r, a_mt), &TyRef(b_r, b_mt)) => {
                let _ = self.relate(&a_r, &b_r)?;
                let _ = self.relate(&a_mt, &b_mt)?;
                None
            },
            (&TyFnDef(a_def_id, a_substs), &TyFnDef(b_def_id, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let a_sig = a.fn_sig(self.tcx);
                    let b_sig = b.fn_sig(self.tcx);
                    let _ = self.relate_item_substs(a_def_id, a_substs, b_substs)?;
                    let _ = self.relate(a_sig.skip_binder(), b_sig.skip_binder())?;
                }

                Some((a_def_id, b_def_id))
            },
            (&TyFnPtr(a_fty), &TyFnPtr(b_fty)) => {
                let _ = self.relate(&a_fty, &b_fty)?;
                None
            },
            (&TyDynamic(a_obj, a_r), &TyDynamic(b_obj, b_r)) => {
                // TODO
                let _ = self.relate(&a_r, &b_r)?;
                if let (Some(a), Some(b)) = (a_obj.principal(), b_obj.principal()) {
                    let _ = self.relate(&a, &b); // TODO: kill this?
                    Some((a.skip_binder().def_id, b.skip_binder().def_id))
                } else {
                    None
                }
            },
            (&TyTuple(as_, _), &TyTuple(bs, _)) => {
                let _ = as_.iter().zip(bs).map(|(a, b)| self.relate(a, b));
                None
            },
            (&TyProjection(a_data), &TyProjection(b_data)) => {
                let _ = self.relate(&a_data, &b_data)?;
                Some((a_data.item_def_id, b_data.item_def_id))
            },
            (&TyAnon(a_def_id, a_substs), &TyAnon(b_def_id, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let _ = ty::relate::relate_substs(self, None, a_substs, b_substs)?;
                }

                Some((a_def_id, b_def_id))
            },
            (&TyInfer(_), _) | (_, &TyInfer(_)) => {
                // As the original function this is ripped off of, we don't handle these cases.
                panic!("var types encountered in Mismatch::tys")
            },
            _ => None,
        };

        if let Some((old_did, new_did)) = matching {
            if !self.id_mapping.contains_id(old_did) && old_did.krate == self.old_crate {
                self.id_mapping.add_internal_item(old_did, new_did);
                self.item_queue.push_back((old_did, new_did));
            }
        }

        Ok(self.tcx.types.err)
    }

    fn regions(&mut self, a: ty::Region<'tcx>, _: ty::Region<'tcx>)
        -> RelateResult<'tcx, ty::Region<'tcx>>
    {
        Ok(a)
    }

    fn binders<T: Relate<'tcx>>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
        -> RelateResult<'tcx, ty::Binder<T>>
    {
        Ok(ty::Binder(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
