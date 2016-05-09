// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Partitioning Codegen Units for Incremental Compilation
//! ======================================================
//!
//! The task of this module is to take the complete set of translation items of
//! a crate and produce a set of codegen units from it, where a codegen unit
//! is a named set of (translation-item, linkage) pairs. That is, this module
//! decides which translation item appears in which codegen units with which
//! linkage. The following paragraphs describe some of the background on the
//! partitioning scheme.
//!
//! The most important opportunity for saving on compilation time with
//! incremental compilation is to avoid re-translating and re-optimizing code.
//! Since the unit of translation and optimization for LLVM is "modules" or, how
//! we call them "codegen units", the particulars of how much time can be saved
//! by incremental compilation are tightly linked to how the output program is
//! partitioned into these codegen units prior to passing it to LLVM --
//! especially because we have to treat codegen units as opaque entities once
//! they are created: There is no way for us to incrementally update an existing
//! LLVM module and so we have to build any such module from scratch if it was
//! affected by some change in the source code.
//!
//! From that point of view it would make sense to maximize the number of
//! codegen units by, for example, putting each function into its own module.
//! That way only those modules would have to be re-compiled that were actually
//! affected by some change, minimizing the number of functions that could have
//! been re-used but just happened to be located in a module that is
//! re-compiled.
//!
//! However, since LLVM optimization does not work across module boundaries,
//! using such a highly granular partitioning would lead to very slow runtime
//! code since it would effectively prohibit inlining and other inter-procedure
//! optimizations. We want to avoid that as much as possible.
//!
//! Thus we end up with a trade-off: The bigger the codegen units, the better
//! LLVM's optimizer can do its work, but also the smaller the compilation time
//! reduction we get from incremental compilation.
//!
//! Ideally, we would create a partitioning such that there are few big codegen
//! units with few interdependencies between them. For now though, we use the
//! following heuristic to determine the partitioning:
//!
//! - There are two codegen units for every source-level module:
//! - One for "stable", that is non-generic, code
//! - One for more "volatile" code, i.e. monomorphized instances of functions
//!   defined in that module
//! - Code for monomorphized instances of functions from external crates gets
//!   placed into every codegen unit that uses that instance.
//!
//! In order to see why this heuristic makes sense, let's take a look at when a
//! codegen unit can get invalidated:
//!
//! 1. The most straightforward case is when the BODY of a function or global
//! changes. Then any codegen unit containing the code for that item has to be
//! re-compiled. Note that this includes all codegen units where the function
//! has been inlined.
//!
//! 2. The next case is when the SIGNATURE of a function or global changes. In
//! this case, all codegen units containing a REFERENCE to that item have to be
//! re-compiled. This is a superset of case 1.
//!
//! 3. The final and most subtle case is when a REFERENCE to a generic function
//! is added or removed somewhere. Even though the definition of the function
//! might be unchanged, a new REFERENCE might introduce a new monomorphized
//! instance of this function which has to be placed and compiled somewhere.
//! Conversely, when removing a REFERENCE, it might have been the last one with
//! that particular set of generic arguments and thus we have to remove it.
//!
//! From the above we see that just using one codegen unit per source-level
//! module is not such a good idea, since just adding a REFERENCE to some
//! generic item somewhere else would invalidate everything within the module
//! containing the generic item. The heuristic above reduces this detrimental
//! side-effect of references a little by at least not touching the non-generic
//! code of the module.
//!
//! As another optimization, monomorphized functions from external crates get
//! some special handling. Since we assume that the definition of such a
//! function changes rather infrequently compared to local items, we can just
//! instantiate external functions in every codegen unit where it is referenced
//! -- without having to fear that doing this will cause a lot of unnecessary
//! re-compilations. If such a reference is added or removed, the codegen unit
//! has to be re-translated anyway.
//! (Note that this only makes sense if external crates actually don't change
//! frequently. For certain multi-crate projects this might not be a valid
//! assumption).
//!
//! A Note on Inlining
//! ------------------
//! As briefly mentioned above, in order for LLVM to be able to inline a
//! function call, the body of the function has to be available in the LLVM
//! module where the call is made. This has a few consequences for partitioning:
//!
//! - The partitioning algorithm has to take care of placing functions into all
//!   codegen units where they should be available for inlining. It also has to
//!   decide on the correct linkage for these functions.
//!
//! - The partitioning algorithm has to know which functions are likely to get
//!   inlined, so it can distribute function instantiations accordingly. Since
//!   there is no way of knowing for sure which functions LLVM will decide to
//!   inline in the end, we apply a heuristic here: Only functions marked with
//!   #[inline] and (as stated above) functions from external crates are
//!   considered for inlining by the partitioner. The current implementation
//!   will not try to determine if a function is likely to be inlined by looking
//!   at the functions definition.
//!
//! Note though that as a side-effect of creating a codegen units per
//! source-level module, functions from the same module will be available for
//! inlining, even when they are not marked #[inline].

use collector::ReferenceMap;
use llvm;
use monomorphize;
use rustc::hir::def_id::DefId;
use rustc::hir::map::DefPathData;
use rustc::ty::TyCtxt;
use rustc::ty::item_path::characteristic_def_id_of_type;
use syntax::parse::token::{self, InternedString};
use trans_item::TransItem;
use util::nodemap::{FnvHashMap, FnvHashSet};

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum InstantiationMode {
    /// This variant indicates that a translation item should be placed in some
    /// codegen unit as a definition and with the given linkage.
    Def(llvm::Linkage),

    /// This variant indicates that only a declaration of some translation item
    /// should be placed in a given codegen unit.
    Decl
}

pub struct CodegenUnit<'tcx> {
    pub name: InternedString,
    pub items: FnvHashMap<TransItem<'tcx>, InstantiationMode>,
}

pub enum PartitioningStrategy {
    /// Generate one codegen unit per source-level module.
    PerModule,

    /// Partition the whole crate into a fixed number of codegen units.
    FixedUnitCount(usize)
}

// Anything we can't find a proper codegen unit for goes into this.
const FALLBACK_CODEGEN_UNIT: &'static str = "__rustc_fallback_codegen_unit";

pub fn partition<'a, 'tcx, I>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              trans_items: I,
                              strategy: PartitioningStrategy,
                              reference_map: &ReferenceMap<'tcx>)
                              -> Vec<CodegenUnit<'tcx>>
    where I: Iterator<Item = TransItem<'tcx>>
{
    // In the first step, we place all regular translation items into their
    // respective 'home' codegen unit. Regular translation items are all
    // functions and statics defined in the local crate.
    let mut initial_partitioning = place_root_translation_items(tcx, trans_items);

    // If the partitioning should produce a fixed count of codegen units, merge
    // until that count is reached.
    if let PartitioningStrategy::FixedUnitCount(count) = strategy {
        merge_codegen_units(&mut initial_partitioning, count, &tcx.crate_name[..]);
    }

    // In the next step, we use the inlining map to determine which addtional
    // translation items have to go into each codegen unit. These additional
    // translation items can be drop-glue, functions from external crates, and
    // local functions the definition of which is marked with #[inline].
    let post_inlining = place_inlined_translation_items(initial_partitioning,
                                                        reference_map);

    // Now we know all *definitions* within all codegen units, thus we can
    // easily determine which declarations need to be placed within each one.
    let post_declarations = place_declarations(post_inlining, reference_map);

    post_declarations.0
}

struct PreInliningPartitioning<'tcx> {
    codegen_units: Vec<CodegenUnit<'tcx>>,
    roots: FnvHashSet<TransItem<'tcx>>,
}

struct PostInliningPartitioning<'tcx>(Vec<CodegenUnit<'tcx>>);
struct PostDeclarationsPartitioning<'tcx>(Vec<CodegenUnit<'tcx>>);

fn place_root_translation_items<'a, 'tcx, I>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                             trans_items: I)
                                             -> PreInliningPartitioning<'tcx>
    where I: Iterator<Item = TransItem<'tcx>>
{
    let mut roots = FnvHashSet();
    let mut codegen_units = FnvHashMap();

    for trans_item in trans_items {
        let is_root = match trans_item {
            TransItem::Static(..) => true,
            TransItem::DropGlue(..) => false,
            TransItem::Fn(_) => !trans_item.is_from_extern_crate(),
        };

        if is_root {
            let characteristic_def_id = characteristic_def_id_of_trans_item(tcx, trans_item);
            let is_volatile = trans_item.is_lazily_instantiated();

            let codegen_unit_name = match characteristic_def_id {
                Some(def_id) => compute_codegen_unit_name(tcx, def_id, is_volatile),
                None => InternedString::new(FALLBACK_CODEGEN_UNIT),
            };

            let make_codegen_unit = || {
                CodegenUnit {
                    name: codegen_unit_name.clone(),
                    items: FnvHashMap(),
                }
            };

            let mut codegen_unit = codegen_units.entry(codegen_unit_name.clone())
                                                .or_insert_with(make_codegen_unit);

            let linkage = match trans_item.explicit_linkage(tcx) {
                Some(explicit_linkage) => explicit_linkage,
                None => {
                    match trans_item {
                        TransItem::Static(..) => llvm::ExternalLinkage,
                        TransItem::DropGlue(..) => unreachable!(),
                        // Is there any benefit to using ExternalLinkage?:
                        TransItem::Fn(..) => llvm::WeakODRLinkage,
                    }
                }
            };

            codegen_unit.items.insert(trans_item,
                                      InstantiationMode::Def(linkage));
            roots.insert(trans_item);
        }
    }

    PreInliningPartitioning {
        codegen_units: codegen_units.into_iter()
                                    .map(|(_, codegen_unit)| codegen_unit)
                                    .collect(),
        roots: roots,
    }
}

fn merge_codegen_units<'tcx>(initial_partitioning: &mut PreInliningPartitioning<'tcx>,
                             target_cgu_count: usize,
                             crate_name: &str) {
    assert!(target_cgu_count >= 1);
    let codegen_units = &mut initial_partitioning.codegen_units;

    // Merge the two smallest codegen units until the target size is reached.
    // Note that "size" is estimated here rather inaccurately as the number of
    // translation items in a given unit. This could be improved on.
    while codegen_units.len() > target_cgu_count {
        // Sort small cgus to the back
        codegen_units.as_mut_slice().sort_by_key(|cgu| -(cgu.items.len() as i64));
        let smallest = codegen_units.pop().unwrap();
        let second_smallest = codegen_units.last_mut().unwrap();

        for (k, v) in smallest.items.into_iter() {
            second_smallest.items.insert(k, v);
        }
    }

    for (index, cgu) in codegen_units.iter_mut().enumerate() {
        cgu.name = numbered_codegen_unit_name(crate_name, index);
    }

    // If the initial partitioning contained less than target_cgu_count to begin
    // with, we won't have enough codegen units here, so add a empty units until
    // we reach the target count
    while codegen_units.len() < target_cgu_count {
        let index = codegen_units.len();
        codegen_units.push(CodegenUnit {
            name: numbered_codegen_unit_name(crate_name, index),
            items: FnvHashMap()
        });
    }

    fn numbered_codegen_unit_name(crate_name: &str, index: usize) -> InternedString {
        token::intern_and_get_ident(&format!("{}.{}", crate_name, index)[..])
    }
}

fn place_inlined_translation_items<'tcx>(initial_partitioning: PreInliningPartitioning<'tcx>,
                                         reference_map: &ReferenceMap<'tcx>)
                                         -> PostInliningPartitioning<'tcx> {
    let mut new_partitioning = Vec::new();

    for codegen_unit in &initial_partitioning.codegen_units[..] {
        // Collect all items that need to be available in this codegen unit
        let mut reachable = FnvHashSet();
        for root in codegen_unit.items.keys() {
            follow_inlining(*root, reference_map, &mut reachable);
        }

        let mut new_codegen_unit = CodegenUnit {
            name: codegen_unit.name.clone(),
            items: FnvHashMap(),
        };

        // Add all translation items that are not already there
        for trans_item in reachable {
            if let Some(instantiation_mode) = codegen_unit.items.get(&trans_item) {
                // This is a root, just copy it over
                new_codegen_unit.items.insert(trans_item, *instantiation_mode);
            } else {
                if initial_partitioning.roots.contains(&trans_item) {
                    // This item will be instantiated in some other codegen unit,
                    // so we just add it here with AvailableExternallyLinkage
                    new_codegen_unit.items.insert(trans_item,
                        InstantiationMode::Def(llvm::AvailableExternallyLinkage));
                } else {
                    // We can't be sure if this will also be instantiated
                    // somewhere else, so we add an instance here with
                    // LinkOnceODRLinkage. That way the item can be discarded if
                    // it's not needed (inlined) after all.
                    new_codegen_unit.items.insert(trans_item,
                        InstantiationMode::Def(llvm::LinkOnceODRLinkage));
                }
            }
        }

        new_partitioning.push(new_codegen_unit);
    }

    return PostInliningPartitioning(new_partitioning);

    fn follow_inlining<'tcx>(trans_item: TransItem<'tcx>,
                             reference_map: &ReferenceMap<'tcx>,
                             visited: &mut FnvHashSet<TransItem<'tcx>>) {
        if !visited.insert(trans_item) {
            return;
        }

        reference_map.with_inlining_candidates(trans_item, |target| {
            follow_inlining(target, reference_map, visited);
        });
    }
}

fn place_declarations<'tcx>(codegen_units: PostInliningPartitioning<'tcx>,
                            reference_map: &ReferenceMap<'tcx>)
                            -> PostDeclarationsPartitioning<'tcx> {
    let PostInliningPartitioning(mut codegen_units) = codegen_units;

    for codegen_unit in codegen_units.iter_mut() {
        let mut declarations = FnvHashSet();

        for (trans_item, _) in &codegen_unit.items {
            for referenced_item in reference_map.get_direct_references_from(*trans_item) {
                if !codegen_unit.items.contains_key(referenced_item) {
                    declarations.insert(*referenced_item);
                }
            }
        }

        codegen_unit.items
                    .extend(declarations.iter()
                                        .map(|trans_item| (*trans_item,
                                                           InstantiationMode::Decl)));
    }

    PostDeclarationsPartitioning(codegen_units)
}

fn characteristic_def_id_of_trans_item<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                                 trans_item: TransItem<'tcx>)
                                                 -> Option<DefId> {
    match trans_item {
        TransItem::Fn(instance) => {
            // If this is a method, we want to put it into the same module as
            // its self-type. If the self-type does not provide a characteristic
            // DefId, we use the location of the impl after all.

            if let Some(self_ty) = instance.substs.self_ty() {
                // This is an implementation of a trait method.
                return characteristic_def_id_of_type(self_ty).or(Some(instance.def));
            }

            if let Some(impl_def_id) = tcx.impl_of_method(instance.def) {
                // This is a method within an inherent impl, find out what the
                // self-type is:
                let impl_self_ty = tcx.lookup_item_type(impl_def_id).ty;
                let impl_self_ty = tcx.erase_regions(&impl_self_ty);
                let impl_self_ty = monomorphize::apply_param_substs(tcx,
                                                                    instance.substs,
                                                                    &impl_self_ty);

                if let Some(def_id) = characteristic_def_id_of_type(impl_self_ty) {
                    return Some(def_id);
                }
            }

            Some(instance.def)
        }
        TransItem::DropGlue(dg) => characteristic_def_id_of_type(dg.ty()),
        TransItem::Static(node_id) => Some(tcx.map.local_def_id(node_id)),
    }
}

fn compute_codegen_unit_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       def_id: DefId,
                                       volatile: bool)
                                       -> InternedString {
    // Unfortunately we cannot just use the `ty::item_path` infrastructure here
    // because we need paths to modules and the DefIds of those are not
    // available anymore for external items.
    let mut mod_path = String::with_capacity(64);

    let def_path = tcx.def_path(def_id);
    mod_path.push_str(&tcx.crate_name(def_path.krate));

    for part in tcx.def_path(def_id)
                   .data
                   .iter()
                   .take_while(|part| {
                        match part.data {
                            DefPathData::Module(..) => true,
                            _ => false,
                        }
                    }) {
        mod_path.push_str("-");
        mod_path.push_str(&part.data.as_interned_str());
    }

    if volatile {
        mod_path.push_str(".volatile");
    }

    return token::intern_and_get_ident(&mod_path[..]);
}
