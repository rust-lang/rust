//! The logic for the second analysis pass collecting mismatched non-public items to match them.
//!
//! Any two items' types found in the same place which are not matched with other items yet are
//! are treated as renamed instances of the same item (as long as they are both unknown to us at
//! the time of analysis). Thus, we may match them up to avoid some false positives.

use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::ty::{Ty, TyCtxt};
use rustc::ty::Visibility::Public;
use rustc::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc::ty::subst::Substs;

use semcheck::mapping::IdMapping;

use std::collections::{HashMap, HashSet, VecDeque};

/// A relation searching for items appearing at the same spot in a type.
///
/// Keeps track of item pairs found that way that correspond to item matchings not yet known.
/// This allows to match up some items that aren't exported, and which possibly even differ in
/// their names across versions.
pub struct MismatchRelation<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    /// The type context used.
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    /// The queue of found item pairings to be processed.
    item_queue: VecDeque<(DefId, DefId)>,
    /// The id mapping to use.
    id_mapping: &'a mut IdMapping,
    /// Type cache holding all old types currently being processed to avoid loops.
    current_old_types: HashSet<Ty<'tcx>>,
    /// Type cache holding all new types currently being processed to avoid loops.
    current_new_types: HashSet<Ty<'tcx>>,
}

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> MismatchRelation<'a, 'gcx, 'tcx> {
    /// Construct a new mismtach type relation.
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>, id_mapping: &'a mut IdMapping) -> Self
    {
        MismatchRelation {
            tcx: tcx,
            item_queue: id_mapping.toplevel_queue(),
            id_mapping: id_mapping,
            current_old_types: Default::default(),
            current_new_types: Default::default(),
        }
    }

    /// Process the next pair of `DefId`s in the queue.
    pub fn process(&mut self) {
        use rustc::hir::def::Def::*;

        while let Some((old_def_id, new_def_id)) = self.item_queue.pop_front() {
            debug!("processing mismatch item pair, remaining: {}", self.item_queue.len());
            debug!("old: {:?}, new: {:?}", old_def_id, new_def_id);
            match self.tcx.describe_def(old_def_id) {
                Some(Trait(_)) | Some(Macro(_, _)) => continue,
                _ => (),
            }

            let old_ty = self.tcx.type_of(old_def_id);
            let new_ty = self.tcx.type_of(new_def_id);
            debug!("relating item pair");
            let _ = self.relate(&old_ty, &new_ty);
        }
    }

    /// Ensure that the pair of given `Substs` is suitable to be related.
    fn check_substs(&self, a_substs: &'tcx Substs<'tcx>, b_substs: &'tcx Substs<'tcx>) -> bool {
        use rustc::ty::subst::UnpackedKind::*;

        for (a, b) in a_substs.iter().zip(b_substs) {
            match (a.unpack(), b.unpack()) {
                (Lifetime(_), Type(_)) | (Type(_), Lifetime(_)) => return false,
                _ => (),
            }
        }

        true
    }
}

impl<'a, 'gcx, 'tcx> TypeRelation<'a, 'gcx, 'tcx> for MismatchRelation<'a, 'gcx, 'tcx> {
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
        debug!("relate: mismatch relation: a: {:?}, b: {:?}", a, b);
        Relate::relate(self, a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        use rustc::ty::TyKind;

        if self.current_old_types.contains(a) || self.current_new_types.contains(b) {
            return Ok(self.tcx.types.err);
        }

        self.current_old_types.insert(a);
        self.current_new_types.insert(b);

        debug!("tys: mismatch relation: a: {:?}, b: {:?}", a, b);
        let matching = match (&a.sty, &b.sty) {
            (&TyKind::Adt(a_def, a_substs), &TyKind::Adt(b_def, b_substs)) => {
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
                        let a_field_ty = field.ty(self.tcx, a_substs);

                        if let Some(b_field) =
                            self.id_mapping
                                .get_new_id(field.did)
                                .and_then(|did| b_fields.get(&did)) {
                            let b_field_ty = b_field.ty(self.tcx, b_substs);

                            let _ = self.relate(&a_field_ty, &b_field_ty)?;
                        }
                    }

                    Some((a_def.did, b_def.did))
                } else {
                    None
                }
            },
            (&TyKind::Array(a_t, _), &TyKind::Array(b_t, _)) |
            (&TyKind::Slice(a_t), &TyKind::Slice(b_t)) => {
                let _ = self.relate(&a_t, &b_t)?;
                None
            },
            (&TyKind::RawPtr(a_mt), &TyKind::RawPtr(b_mt)) => {
                let _ = self.relate(&a_mt, &b_mt)?;
                None
            },
            (&TyKind::Ref(a_r, a_ty, _), &TyKind::Ref(b_r, b_ty, _)) => {
                let _ = self.relate(&a_r, &b_r)?;
                let _ = self.relate(&a_ty, &b_ty)?;
                None
            },
            (&TyKind::FnDef(a_def_id, a_substs), &TyKind::FnDef(b_def_id, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let a_sig = a.fn_sig(self.tcx);
                    let b_sig = b.fn_sig(self.tcx);
                    let _ = self.relate_item_substs(a_def_id, a_substs, b_substs)?;
                    let _ = self.relate(a_sig.skip_binder(), b_sig.skip_binder())?;
                }

                Some((a_def_id, b_def_id))
            },
            (&TyKind::FnPtr(a_fty), &TyKind::FnPtr(b_fty)) => {
                let _ = self.relate(&a_fty, &b_fty)?;
                None
            },
            (&TyKind::Dynamic(a_obj, a_r), &TyKind::Dynamic(b_obj, b_r)) => {
                let _ = self.relate(&a_r, &b_r)?;

                match (a_obj.principal(), b_obj.principal()) {
                    (Some(a), Some(b)) if self.check_substs(a.skip_binder().substs,
                                                            b.skip_binder().substs) => {
                        let _ = self.relate(&a.skip_binder().substs, &b.skip_binder().substs)?;
                        Some((a.skip_binder().def_id, b.skip_binder().def_id))
                    },
                    _ => None,
                }
            },
            (&TyKind::Tuple(as_), &TyKind::Tuple(bs)) => {
                let _ = as_.iter().zip(bs).map(|(a, b)| self.relate(a, b));
                None
            },
            (&TyKind::Projection(a_data), &TyKind::Projection(b_data)) => {
                let _ = self.relate(&a_data, &b_data)?;
                Some((a_data.item_def_id, b_data.item_def_id))
            },
            (&TyKind::Opaque(a_def_id, a_substs), &TyKind::Opaque(b_def_id, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let _ = ty::relate::relate_substs(self, None, a_substs, b_substs)?;
                }

                Some((a_def_id, b_def_id))
            },
            (&TyKind::Infer(_), _) | (_, &TyKind::Infer(_)) => {
                // As the original function this is ripped off of, we don't handle these cases.
                panic!("var types encountered in MismatchRelation::tys")
            },
            _ => None,
        };

        self.current_old_types.remove(a);
        self.current_new_types.remove(b);

        if let Some((old_def_id, new_def_id)) = matching {
            if !self.id_mapping.contains_old_id(old_def_id) &&
                    self.id_mapping.in_old_crate(old_def_id) {
                self.id_mapping.add_internal_item(old_def_id, new_def_id);
                self.item_queue.push_back((old_def_id, new_def_id));
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
        Ok(ty::Binder::bind(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
