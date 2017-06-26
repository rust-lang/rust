use semcheck::changes::BinaryChangeType;
use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::ChangeSet;

use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::ty::{Ty, TyCtxt};
use rustc::ty::Visibility::Public;
use rustc::ty::error::TypeError;
use rustc::ty::fold::{BottomUpFolder, TypeFoldable};
use rustc::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc::ty::subst::{Subst, Substs};

use std::collections::{HashMap, HashSet, VecDeque};

/// A mapping from old to new `DefId`s, as well as exports.
///
/// Exports and simple `DefId` mappings are kept separate to record both kinds of correspondence
/// losslessly. The *access* to the stored data happens through the same API, however.
#[derive(Default)]
struct IdMapping {
    /// Toplevel items' old `DefId` mapped to new `DefId`, as well as old and new exports.
    toplevel_mapping: HashMap<DefId, (DefId, Export, Export)>,
    /// Other item's old `DefId` mapped to new `DefId`.
    mapping: HashMap<DefId, DefId>,
}

impl IdMapping {
    /// Register two exports representing the same item across versions.
    pub fn add_export(&mut self, old: Export, new: Export) -> bool {
        self.toplevel_mapping
            .insert(old.def.def_id(), (new.def.def_id(), old, new))
            .is_some()
    }

    /// Add any other item pair's old and new `DefId`s.
    pub fn add_item(&mut self, old: DefId, new: DefId) {
        self.mapping.insert(old, new);
    }

    /// Get the new `DefId` associated with the given old one.
    pub fn get_new_id(&self, old: DefId) -> DefId {
        if let Some(new) = self.toplevel_mapping.get(&old) {
            new.0
        } else {
            self.mapping[&old]
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
enum Namespace {
    Type,
    Value,
    Macro,
    Err,
}

fn get_namespace(def: &Def) -> Namespace {
    use rustc::hir::def::Def::*;

    match *def {
        Mod(_) |
        Struct(_) |
        Union(_) |
        Enum(_) |
        Variant(_) |
        Trait(_) |
        TyAlias(_) |
        AssociatedTy(_) |
        PrimTy(_) |
        TyParam(_) |
        SelfTy(_, _) => Namespace::Type,
        Fn(_) |
        Const(_) |
        Static(_, _) |
        StructCtor(_, _) |
        VariantCtor(_, _) |
        Method(_) |
        AssociatedConst(_) |
        Local(_) |
        Upvar(_, _, _) |
        Label(_) => Namespace::Value,
        Macro(_, _) => Namespace::Macro,
        GlobalAsm(_) |
        Err => Namespace::Err,
    }
}

/// Traverse the two root modules in an interleaved manner.
///
/// Match up pairs of modules from the two crate versions and compare for changes.
/// Matching children get processed in the same fashion.
// TODO: describe the passes we do.
pub fn traverse_modules(tcx: TyCtxt, old: DefId, new: DefId) -> ChangeSet {
    use rustc::hir::def::Def::*;

    let cstore = &tcx.sess.cstore;
    let mut changes = ChangeSet::default();
    let mut id_mapping = IdMapping::default();
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((old, new, Public, Public));

    while let Some((old_did, new_did, old_vis, new_vis)) = mod_queue.pop_front() {
        let mut c_old = cstore.item_children(old_did, tcx.sess);
        let mut c_new = cstore.item_children(new_did, tcx.sess);

        // TODO: refactor this to avoid storing tons of `Namespace` values.
        for child in c_old.drain(..) {
            let key = (get_namespace(&child.def), child.ident.name);
            children.entry(key).or_insert((None, None)).0 = Some(child);
        }

        for child in c_new.drain(..) {
            let key = (get_namespace(&child.def), child.ident.name);
            children.entry(key).or_insert((None, None)).1 = Some(child);
        }

        for (_, items) in children.drain() {
            match items {
                (Some(Export { def: Mod(o), .. }), Some(Export { def: Mod(n), .. })) => {
                    if visited.insert((o, n)) {
                        let o_vis = if old_vis == Public {
                            cstore.visibility(o)
                        } else {
                            old_vis
                        };
                        let n_vis = if new_vis == Public {
                            cstore.visibility(n)
                        } else {
                            new_vis
                        };

                        if o_vis != n_vis {
                            // TODO: ugly
                            changes.new_binary(items.0.unwrap(), items.1.unwrap(), true);

                            if o_vis == Public && n_vis != Public {
                                changes.add_binary(ItemMadePrivate, o, None);
                            } else if o_vis != Public && n_vis == Public {
                                changes.add_binary(ItemMadePublic, o, None);
                            }
                        }

                        mod_queue.push_back((o, n, o_vis, n_vis));
                    }
                }
                (Some(o), Some(n)) => {
                    if !id_mapping.add_export(o, n) {
                        let o_def_id = o.def.def_id();
                        let n_def_id = n.def.def_id();
                        let o_vis = if old_vis == Public {
                            cstore.visibility(o_def_id)
                        } else {
                            old_vis
                        };
                        let n_vis = if new_vis == Public {
                            cstore.visibility(n_def_id)
                        } else {
                            new_vis
                        };

                        let output = o_vis == Public || n_vis == Public;
                        changes.new_binary(o, n, output);

                        if o_vis == Public && n_vis != Public {
                            changes.add_binary(ItemMadePrivate, o_def_id, None);
                        } else if o_vis != Public && n_vis == Public {
                            changes.add_binary(ItemMadePublic, o_def_id, None);
                        }

                        // TODO: extend as needed.
                        match (o.def, n.def) {
                            // (matching) things we don't care about (for now)
                            (Mod(_), Mod(_)) |
                            (Trait(_), Trait(_)) |
                            (AssociatedTy(_), AssociatedTy(_)) |
                            (PrimTy(_), PrimTy(_)) |
                            (TyParam(_), TyParam(_)) |
                            (SelfTy(_, _), SelfTy(_, _)) |
                            (Fn(_), Fn(_)) |
                            (StructCtor(_, _), StructCtor(_, _)) |
                            (VariantCtor(_, _), VariantCtor(_, _)) |
                            (Method(_), Method(_)) |
                            (AssociatedConst(_), AssociatedConst(_)) |
                            (Local(_), Local(_)) |
                            (Upvar(_, _, _), Upvar(_, _, _)) |
                            (Label(_), Label(_)) |
                            (GlobalAsm(_), GlobalAsm(_)) |
                            (Macro(_, _), Macro(_, _)) |
                            (Variant(_), Variant(_)) |
                            (Const(_), Const(_)) |
                            (Static(_, _), Static(_, _)) |
                            (Err, Err) => {},
                            (TyAlias(_), TyAlias(_)) => {
                                let mut generics_changes =
                                    diff_generics(tcx, o_def_id, n_def_id);
                                for change_type in generics_changes.drain(..) {
                                    changes.add_binary(change_type, o_def_id, None);
                                }
                            },
                            // ADTs for now
                            (Struct(_), Struct(_)) |
                            (Union(_), Union(_)) |
                            (Enum(_), Enum(_)) => {
                                let mut generics_changes =
                                    diff_generics(tcx, o_def_id, n_def_id);
                                for change_type in generics_changes.drain(..) {
                                    changes.add_binary(change_type, o_def_id, None);
                                }

                                diff_adts(&mut changes, &mut id_mapping, tcx, o, n);
                            },
                            // non-matching item pair - register the difference and abort
                            _ => {
                                changes.add_binary(KindDifference, o_def_id, None);
                            },
                        }
                    }
                }
                (Some(o), None) => {
                    if old_vis == Public && cstore.visibility(o.def.def_id()) == Public {
                        changes.new_removal(o);
                    }
                }
                (None, Some(n)) => {
                    if new_vis == Public && cstore.visibility(n.def.def_id()) == Public {
                        changes.new_addition(n);
                    }
                }
                (None, None) => unreachable!(),
            }
        }
    }

    for &(_, old, new) in id_mapping.toplevel_mapping.values() {
        let old_did = old.def.def_id();
        let new_did = new.def.def_id();

        {
            let mut mismatch = Mismatch {
                tcx: tcx,
                toplevel_mapping: &id_mapping.toplevel_mapping,
                mapping: &mut id_mapping.mapping,
            };

            let _ = mismatch.tys(tcx.type_of(old_did), tcx.type_of(new_did));
        }

        diff_bounds(&mut changes, tcx, old_did, new_did);
    }

    for &(_, old, new) in id_mapping.toplevel_mapping.values() {
        diff_items(&mut changes, &id_mapping, tcx, old, new);
    }

    changes
}

fn diff_bounds(_changes: &mut ChangeSet, _tcx: TyCtxt, _old: DefId, _new: DefId)
    -> (Vec<BinaryChangeType>, Vec<(DefId, DefId)>)
{
    let res = Default::default();

    // let old_preds = tcx.predicates_of(old).predicates;
    // let new_preds = tcx.predicates_of(new).predicates;

    res
}

struct Mismatch<'a, 'gcx: 'a + 'tcx, 'tcx: 'a, A: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    toplevel_mapping: &'a HashMap<DefId, A>,
    mapping: &'a mut HashMap<DefId, DefId>,
}

impl<'a, 'gcx, 'tcx, A: 'a> TypeRelation<'a, 'gcx, 'tcx> for Mismatch<'a, 'gcx, 'tcx, A> {
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
            (&TyFnDef(a_did, _, _), &TyFnDef(b_did, _, _)) => Some((a_did, b_did)),
            (&TyClosure(a_did, _), &TyClosure(b_did, _)) => Some((a_did, b_did)),
            (&TyProjection(a_proj), &TyProjection(b_proj)) =>
                Some((a_proj.trait_ref.def_id, b_proj.trait_ref.def_id)),
            (&TyAnon(a_did, _), &TyAnon(b_did, _)) => Some((a_did, b_did)),
            _ => {
                None
            },
        };

        if let Some((old_did, new_did)) = matching {
            if !self.toplevel_mapping.contains_key(&old_did) &&
                !self.mapping.contains_key(&old_did)
            {
                // println!("adding mapping: {:?} => {:?}", old_did, new_did);
                self.mapping.insert(old_did, new_did);
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
fn relate_tys_mismatch<'a, 'gcx, 'tcx, R>(relation: &mut R, a: Ty<'tcx>, b: Ty<'tcx>)
    -> RelateResult<'tcx, Ty<'tcx>>
    where R: TypeRelation<'a, 'gcx, 'tcx>, 'gcx: 'a+'tcx, 'tcx: 'a
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
        (&TyDynamic(_, _), &TyDynamic(_, _)) => {
            // TODO: decide whether this is needed
            /*let region_bound = relation.with_cause(Cause::ExistentialRegionBound,
                                                   |relation| {
                                                       relation.relate(a_region, b_region)
                                                   })?;
            Ok(tcx.mk_dynamic(relation.relate(a_obj, b_obj)?, region_bound))*/
            Err(TypeError::Mismatch)
        },
        (&TyRawPtr(ref a_mt), &TyRawPtr(ref b_mt)) => {
            let mt = relation.relate(a_mt, b_mt)?;
            Ok(tcx.mk_ptr(mt))
        },
        (&TyRef(ref a_r, ref a_mt), &TyRef(ref b_r, ref b_mt)) => {
            let r = relation.relate(a_r, b_r)?;
            let mt = relation.relate(a_mt, b_mt)?;
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
            } else if !(as_.is_empty() || bs.is_empty()) {
                Err(TypeError::Mismatch)
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

/// Given two ADT items, perform *structural* checks.
///
/// This establishes the needed correspondence relationship between non-toplevel items.
/// For instance, struct fields, enum variants etc. are matched up against each other here.
fn diff_adts(changes: &mut ChangeSet,
             id_mapping: &mut IdMapping,
             tcx: TyCtxt,
             old: Export,
             new: Export) {
    use rustc::hir::def::Def::*;

    let old_def_id = old.def.def_id();
    let new_def_id = new.def.def_id();

    let (old_def, new_def) = match (old.def, new.def) {
        (Struct(_), Struct(_)) |
        (Union(_), Union(_)) |
        (Enum(_), Enum(_)) => (tcx.adt_def(old_def_id), tcx.adt_def(new_def_id)),
        _ => return,
    };

    let mut variants = HashMap::new();
    let mut fields = HashMap::new();

    for variant in &old_def.variants {
        variants.entry(variant.name).or_insert((None, None)).0 = Some(variant);
    }

    for variant in &new_def.variants {
        variants.entry(variant.name).or_insert((None, None)).1 = Some(variant);
    }

    for (_, items) in variants.drain() {
        match items {
            (Some(old), Some(new)) => {
                id_mapping.add_item(old.did, new.did);

                for field in &old.fields {
                    fields.entry(field.name).or_insert((None, None)).0 = Some(field);
                }

                for field in &new.fields {
                    fields.entry(field.name).or_insert((None, None)).1 = Some(field);
                }

                let mut total_private = true;
                let mut total_public = true;

                for items2 in fields.values() {
                    if let Some(o) = items2.0 {
                        let public = o.vis == Public;
                        total_public &= public;
                        total_private &= !public;
                    }
                }

                if old.ctor_kind != new.ctor_kind {
                    let c = VariantStyleChanged {
                        now_struct: new.ctor_kind == CtorKind::Fictive,
                        total_private: total_private,
                    };
                    changes.add_binary(c, old_def_id, Some(tcx.def_span(old.did)));
                }

                for (_, items2) in fields.drain() {
                    match items2 {
                        (Some(o), Some(n)) => {
                            id_mapping.add_item(o.did, n.did);

                            if o.vis != Public && n.vis == Public {
                                changes.add_binary(ItemMadePublic,
                                                   old_def_id,
                                                   Some(tcx.def_span(o.did)));
                            } else if o.vis == Public && n.vis != Public {
                                changes.add_binary(ItemMadePrivate,
                                                   old_def_id,
                                                   Some(tcx.def_span(o.did)));
                            }
                        },
                        (Some(o), None) => {
                            let c = VariantFieldRemoved {
                                public: o.vis == Public,
                                total_public: total_public
                            };
                            changes.add_binary(c, old_def_id, Some(tcx.def_span(o.did)));
                        },
                        (None, Some(n)) => {
                            let c = VariantFieldAdded {
                                public: n.vis == Public,
                                total_public: total_public
                            };
                            changes.add_binary(c, old_def_id, Some(tcx.def_span(n.did)));
                        },
                        (None, None) => unreachable!(),
                    }
                }
            },
            (Some(old), None) => {
                changes.add_binary(VariantRemoved, old_def_id, Some(tcx.def_span(old.did)));
            },
            (None, Some(new)) => {
                changes.add_binary(VariantAdded, old_def_id, Some(tcx.def_span(new.did)));
            },
            (None, None) => unreachable!(),
        }
    }
}

/// Given two items, perform *non-structural* checks.
///
/// This encompasses all checks for type and requires that the structural checks have already
/// been performed.
fn diff_items(changes: &mut ChangeSet,
              id_mapping: &IdMapping,
              tcx: TyCtxt,
              old: Export,
              new: Export) {
    if !changes.item_breaking(old.def.def_id()) {
        let mut type_changes = diff_types(id_mapping, tcx, old.def.def_id(), new.def.def_id());

        for change_type in type_changes.drain(..) {
            changes.add_binary(change_type, old.def.def_id(), None);
        }
    }
}


/// Given two items, compare their types for equality.
fn diff_types(id_mapping: &IdMapping, tcx: TyCtxt, old: DefId, new: DefId)
    -> Vec<BinaryChangeType>
{
    use rustc::ty::AdtDef;
    use rustc::ty::TypeVariants::*;

    let mut res = Vec::new();

    let new_ty = tcx.type_of(new);
    let old_ty = tcx.type_of(old).subst(tcx, Substs::identity_for_item(tcx, new));

    let old_ty_cmp = old_ty.fold_with(&mut BottomUpFolder { tcx: tcx, fldop: |ty| {
        match ty.sty {
            TyAdt(&AdtDef { ref did, .. }, substs) => {
                let new_did = id_mapping.get_new_id(*did);
                let new_adt = tcx.adt_def(new_did);
                tcx.mk_adt(new_adt, substs)
            },
            /* TyDynamic(predicates, region) => {

            }, TyClosure, TyRef (because of region?), TyProjection, TyAnon
            TyProjection(projection_ty) => {

            }, */
            _ => ty,
        }
    }});

    // println!("old_ty: {:?}", old_ty_cmp);
    // println!("new_ty: {:?}", new_ty);

    if let Result::Err(err) = tcx.global_tcx().infer_ctxt()
        .enter(|infcx| infcx.can_eq(tcx.param_env(new), old_ty_cmp, new_ty))
    {
        // println!("diff: {}", err);
        res.push(FieldTypeChanged(format!("{}", err))); // FIXME: this is obv a terrible hack
    }

    res
}

/// Given two items, compare their type and region parameter sets.
fn diff_generics(tcx: TyCtxt, old: DefId, new: DefId) -> Vec<BinaryChangeType> {
    use std::cmp::max;

    let mut ret = Vec::new();

    let old_gen = tcx.generics_of(old);
    let new_gen = tcx.generics_of(new);

    for i in 0..max(old_gen.regions.len(), new_gen.regions.len()) {
        match (old_gen.regions.get(i), new_gen.regions.get(i)) {
            (Some(_ /* old_region */), None) => {
                ret.push(RegionParameterRemoved);
            },
            (None, Some(_ /* new_region */)) => {
                ret.push(RegionParameterAdded);
            },
            (Some(_), Some(_)) => (),
            (None, None) => unreachable!(),
        }
    }

    for i in 0..max(old_gen.types.len(), new_gen.types.len()) {
        match (old_gen.types.get(i), new_gen.types.get(i)) {
            (Some(old_type), Some(new_type)) => {
                if old_type.has_default && !new_type.has_default {
                    // TODO: major for sure
                    ret.push(TypeParameterRemoved { defaulted: true });
                    ret.push(TypeParameterAdded { defaulted: false });
                } else if !old_type.has_default && new_type.has_default {
                    // TODO: minor, I guess?
                    ret.push(TypeParameterRemoved { defaulted: false });
                    ret.push(TypeParameterAdded { defaulted: true });
                }
            },
            (Some(old_type), None) => {
                ret.push(TypeParameterRemoved { defaulted: old_type.has_default });
            },
            (None, Some(new_type)) => {
                ret.push(TypeParameterAdded { defaulted: new_type.has_default });
            },
            (None, None) => unreachable!(),
        }
    }

    ret
}
