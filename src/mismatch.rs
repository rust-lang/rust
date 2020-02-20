//! The logic for the second analysis pass collecting mismatched non-public items to match them.
//!
//! Any two items' types found in the same place which are not matched with other items yet are
//! are treated as renamed instances of the same item (as long as they are both unknown to us at
//! the time of analysis). Thus, we may match them up to avoid some false positives.

use crate::mapping::IdMapping;
use log::debug;
use rustc_hir::def::{DefKind, Res};
use rustc_middle::ty::{
    self,
    relate::{Relate, RelateResult, TypeRelation},
    subst::SubstsRef,
    ParamEnv, Ty, TyCtxt,
    Visibility::Public,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// A relation searching for items appearing at the same spot in a type.
///
/// Keeps track of item pairs found that way that correspond to item matchings not yet known.
/// This allows to match up some items that aren't exported, and which possibly even differ in
/// their names across versions.
#[cfg_attr(feature = "cargo-clippy", allow(clippy::module_name_repetitions))]
pub struct MismatchRelation<'a, 'tcx> {
    /// The type context used.
    tcx: TyCtxt<'tcx>,
    /// The queue of found item pairings to be processed.
    item_queue: VecDeque<(Res, Res)>,
    /// The id mapping to use.
    id_mapping: &'a mut IdMapping,
    /// Type cache holding all old types currently being processed to avoid loops.
    current_old_types: HashSet<Ty<'tcx>>,
    /// Type cache holding all new types currently being processed to avoid loops.
    current_new_types: HashSet<Ty<'tcx>>,
}

impl<'a, 'tcx> MismatchRelation<'a, 'tcx> {
    /// Construct a new mismtach type relation.
    pub fn new(tcx: TyCtxt<'tcx>, id_mapping: &'a mut IdMapping) -> Self {
        Self {
            tcx,
            item_queue: id_mapping.toplevel_queue(),
            id_mapping,
            current_old_types: HashSet::default(),
            current_new_types: HashSet::default(),
        }
    }

    /// Process the next pair of `DefId`s in the queue.
    pub fn process(&mut self) {
        // use rustc_middle::hir::def::DefKind::*;

        while let Some((old_res, new_res)) = self.item_queue.pop_front() {
            debug!(
                "processing mismatch item pair, remaining: {}",
                self.item_queue.len()
            );
            debug!("old: {:?}, new: {:?}", old_res, new_res);

            // FIXME: this is odd, see if we can lift the restriction on traits
            let (old_def_id, new_def_id) = match (old_res, new_res) {
                (Res::Def(k1, o), Res::Def(k2, n)) => {
                    match k1 {
                        DefKind::Trait | DefKind::Macro(_) => continue,
                        _ => (),
                    };

                    match k2 {
                        DefKind::Trait | DefKind::Macro(_) => continue,
                        _ => (),
                    };

                    (o, n)
                }
                _ => continue,
            };

            let old_ty = self.tcx.type_of(old_def_id);
            let new_ty = self.tcx.type_of(new_def_id);
            debug!("relating item pair");
            let _ = self.relate(&old_ty, &new_ty);
        }
    }

    /// Ensure that the pair of given `SubstsRef`s is suitable to be related.
    fn check_substs(&self, a_substs: SubstsRef<'tcx>, b_substs: SubstsRef<'tcx>) -> bool {
        use rustc_middle::ty::subst::GenericArgKind::*;

        for (a, b) in a_substs.iter().zip(b_substs) {
            match (a.unpack(), b.unpack()) {
                (Lifetime(_), Type(_)) | (Type(_), Lifetime(_)) => return false,
                _ => (),
            }
        }

        true
    }
}

impl<'a, 'tcx> TypeRelation<'tcx> for MismatchRelation<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn param_env(&self) -> ParamEnv<'tcx> {
        ParamEnv::empty()
    }

    fn tag(&self) -> &'static str {
        "Mismatch"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _: ty::Variance,
        a: &T,
        b: &T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    fn relate<T: Relate<'tcx>>(&mut self, a: &T, b: &T) -> RelateResult<'tcx, T> {
        debug!("relate: mismatch relation: a: {:?}, b: {:?}", a, b);
        Relate::relate(self, a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        use rustc_middle::ty::TyKind;

        if self.current_old_types.contains(a) || self.current_new_types.contains(b) {
            return Ok(self.tcx.types.err);
        }

        self.current_old_types.insert(a);
        self.current_new_types.insert(b);

        debug!("tys: mismatch relation: a: {:?}, b: {:?}", a, b);
        let matching = match (&a.kind, &b.kind) {
            (&TyKind::Adt(a_def, a_substs), &TyKind::Adt(b_def, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let _ = self.relate_item_substs(a_def.did, a_substs, b_substs)?;
                    let a_adt = self.tcx.adt_def(a_def.did);
                    let b_adt = self.tcx.adt_def(b_def.did);

                    let b_fields: HashMap<_, _> = b_adt.all_fields().map(|f| (f.did, f)).collect();

                    for field in a_adt.all_fields().filter(|f| f.vis == Public) {
                        let a_field_ty = field.ty(self.tcx, a_substs);

                        if let Some(b_field) = self
                            .id_mapping
                            .get_new_id(field.did)
                            .and_then(|did| b_fields.get(&did))
                        {
                            let b_field_ty = b_field.ty(self.tcx, b_substs);

                            let _ = self.relate(&a_field_ty, &b_field_ty)?;
                        }
                    }

                    let a = if a_def.is_struct() {
                        Res::Def(DefKind::Struct, a_def.did)
                    } else if a_def.is_union() {
                        Res::Def(DefKind::Union, a_def.did)
                    } else {
                        Res::Def(DefKind::Enum, a_def.did)
                    };

                    let b = if b_def.is_struct() {
                        Res::Def(DefKind::Struct, b_def.did)
                    } else if b_def.is_union() {
                        Res::Def(DefKind::Union, b_def.did)
                    } else {
                        Res::Def(DefKind::Enum, b_def.did)
                    };

                    Some((a, b))
                } else {
                    None
                }
            }
            (&TyKind::Array(a_t, _), &TyKind::Array(b_t, _))
            | (&TyKind::Slice(a_t), &TyKind::Slice(b_t)) => {
                let _ = self.relate(&a_t, &b_t)?;
                None
            }
            (&TyKind::RawPtr(a_mt), &TyKind::RawPtr(b_mt)) => {
                let _ = self.relate(&a_mt, &b_mt)?;
                None
            }
            (&TyKind::Ref(a_r, a_ty, _), &TyKind::Ref(b_r, b_ty, _)) => {
                let _ = self.relate(&a_r, &b_r)?;
                let _ = self.relate(&a_ty, &b_ty)?;
                None
            }
            (&TyKind::FnDef(a_def_id, a_substs), &TyKind::FnDef(b_def_id, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let a_sig = a.fn_sig(self.tcx);
                    let b_sig = b.fn_sig(self.tcx);
                    let _ = self.relate_item_substs(a_def_id, a_substs, b_substs)?;
                    let _ = self.relate(a_sig.skip_binder(), b_sig.skip_binder())?;
                }

                let a = Res::Def(DefKind::Fn, a_def_id);
                let b = Res::Def(DefKind::Fn, b_def_id);

                Some((a, b))
            }
            (&TyKind::FnPtr(a_fty), &TyKind::FnPtr(b_fty)) => {
                let _ = self.relate(&a_fty, &b_fty)?;
                None
            }
            (&TyKind::Dynamic(a_obj, a_r), &TyKind::Dynamic(b_obj, b_r)) => {
                let _ = self.relate(&a_r, &b_r)?;
                let a = a_obj.principal();
                let b = b_obj.principal();

                if let (Some(a), Some(b)) = (a, b) {
                    if self.check_substs(a.skip_binder().substs, b.skip_binder().substs) {
                        let _ = self.relate(&a.skip_binder().substs, &b.skip_binder().substs)?;
                        let a = Res::Def(DefKind::Trait, a.skip_binder().def_id);
                        let b = Res::Def(DefKind::Trait, b.skip_binder().def_id);
                        Some((a, b))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            (&TyKind::Tuple(as_), &TyKind::Tuple(bs)) => {
                let _ = as_.iter().zip(bs).map(|(a, b)| self.relate(a, b));
                None
            }
            (&TyKind::Projection(a_data), &TyKind::Projection(b_data)) => {
                let _ = self.relate(&a_data, &b_data)?;

                let a = Res::Def(DefKind::AssocTy, a_data.item_def_id);
                let b = Res::Def(DefKind::AssocTy, b_data.item_def_id);

                Some((a, b))
            }
            (&TyKind::Opaque(_a_def_id, a_substs), &TyKind::Opaque(_b_def_id, b_substs)) => {
                if self.check_substs(a_substs, b_substs) {
                    let _ = ty::relate::relate_substs(self, None, a_substs, b_substs)?;
                }

                // TODO: we are talking impl trait here, so we can build a Res for that or the
                // associated type
                // Some((a_def_id, b_def_id))
                None
            }
            (&TyKind::Infer(_), _) | (_, &TyKind::Infer(_)) => {
                // As the original function this is ripped off of, we don't handle these cases.
                panic!("var types encountered in MismatchRelation::tys")
            }
            _ => None,
        };

        self.current_old_types.remove(a);
        self.current_new_types.remove(b);

        if let Some((old, new)) = matching {
            let old_def_id = old.def_id();
            let new_def_id = new.def_id();

            if !self.id_mapping.contains_old_id(old_def_id)
                && self.id_mapping.in_old_crate(old_def_id)
            {
                self.id_mapping.add_internal_item(old_def_id, new_def_id);
                self.item_queue.push_back((old, new));
            }
        }

        Ok(self.tcx.types.err)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        _: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        Ok(a)
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        _: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        Ok(a) // TODO
    }

    fn binders<T: Relate<'tcx>>(
        &mut self,
        a: &ty::Binder<T>,
        b: &ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>> {
        Ok(ty::Binder::bind(
            self.relate(a.skip_binder(), b.skip_binder())?,
        ))
    }
}
