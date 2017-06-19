use semcheck::changes::BinaryChangeType;
use semcheck::changes::BinaryChangeType::*;
use semcheck::changes::ChangeSet;

use rustc::hir::def::Def;
use rustc::hir::def::Def::*;
use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::ty::TyCtxt;
use rustc::ty::fold::{BottomUpFolder, TypeFoldable};
use rustc::ty::subst::{Subst, Substs};
use rustc::ty::Visibility::Public;

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
    let cstore = &tcx.sess.cstore;
    let mut changes = ChangeSet::default();
    let mut id_mapping = IdMapping::default();
    let mut visited = HashSet::new();
    let mut children = HashMap::new();
    let mut mod_queue = VecDeque::new();

    mod_queue.push_back((old, new));

    while let Some((old_did, new_did)) = mod_queue.pop_front() {
        let mut c_old = cstore.item_children(old_did, tcx.sess);
        let mut c_new = cstore.item_children(new_did, tcx.sess);

        // TODO: refactor this to avoid storing tons of `Namespace` values.
        for child in c_old
                .drain(..)
                .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let key = (get_namespace(&child.def), child.ident.name);
            children.entry(key).or_insert((None, None)).0 = Some(child);
        }

        for child in c_new
                .drain(..)
                .filter(|c| cstore.visibility(c.def.def_id()) == Public) {
            let key = (get_namespace(&child.def), child.ident.name);
            children.entry(key).or_insert((None, None)).1 = Some(child);
        }

        for (_, items) in children.drain() {
            match items {
                (Some(Export { def: Mod(o), .. }), Some(Export { def: Mod(n), .. })) => {
                    if !visited.insert((o, n)) {
                        mod_queue.push_back((o, n));
                    }
                }
                (Some(o), Some(n)) => {
                    if !id_mapping.add_export(o, n) {
                        changes.new_binary(o, n);
                        diff_item_structures(&mut changes, &mut id_mapping, tcx, o, n);
                    }
                }
                (Some(old), None) => {
                    changes.new_removal(old);
                }
                (None, Some(new)) => {
                    changes.new_addition(new);
                }
                (None, None) => unreachable!(),
            }
        }
    }

    for &(_, old, new) in id_mapping.toplevel_mapping.values() {
        diff_items(&mut changes, &id_mapping, tcx, old, new);
    }

    changes
}

/// Given two items, perform *structural* checks.
///
/// This establishes the needed correspondence relationship between non-toplevel items.
/// For instance, struct fields, enum variants etc. are matched up against each other here.
///
/// If the two items can't be meaningfully compared because they are of different kinds,
/// we return that difference directly.
fn diff_item_structures(changes: &mut ChangeSet,
                        id_mapping: &mut IdMapping,
                        tcx: TyCtxt,
                        old: Export,
                        new: Export) {
    let old_def_id = old.def.def_id();
    let new_def_id = new.def.def_id();

    let mut generics_changes = diff_generics(tcx, old_def_id, new_def_id);
    for change_type in generics_changes.drain(..) {
        changes.add_binary(change_type, old_def_id, None);
    }

    // TODO: crude dispatching logic for now (needs totality etc).
    match (old.def, new.def) {
        (TyAlias(_), TyAlias(_)) => return,
        (Struct(_), Struct(_)) |
        (Union(_), Union(_)) |
        (Enum(_), Enum(_)) |
        (Trait(_), Trait(_)) |
        (Fn(_), Fn(_)) |
        (Const(_), Const(_)) |
        (Static(_, _), Static(_, _)) |
        (Method(_), Method(_)) |
        (Macro(_, _), Macro(_, _)) => {},
        _ => {
            // No match - so we don't need to look further.
            changes.add_binary(KindDifference, old_def_id, None);
            return;
        },
    }

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

                for (_, items2) in fields.drain() {
                    // TODO: visibility
                    match items2 {
                        (Some(o), Some(n)) => {
                            id_mapping.add_item(o.did, n.did);
                        },
                        (Some(o), None) => {
                            changes.add_binary(VariantFieldRemoved,
                                               old_def_id,
                                               Some(tcx.def_span(o.did)));
                        },
                        (None, Some(n)) => {
                            changes.add_binary(VariantFieldAdded,
                                               old_def_id,
                                               Some(tcx.def_span(n.did)));
                        },
                        (None, None) => unreachable!(),
                    }
                }
            },
            (Some(old), None) => {
                changes.add_binary(EnumVariantRemoved, old_def_id, Some(tcx.def_span(old.did)));
            },
            (None, Some(new)) => {
                changes.add_binary(EnumVariantAdded, old_def_id, Some(tcx.def_span(new.did)));
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
                tcx.mk_adt(new_adt, substs) // TODO: error if mismatch?
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
