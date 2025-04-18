//! Validity checking for weak lang items

use std::iter;

use rustc_attr_parsing::{EIIDecl, EIIImpl};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CRATE_DEF_ID, CrateNum, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::definitions::DisambiguatorState;
use rustc_middle::middle::eii::EiiMapping;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::CrateType;
use rustc_span::Symbol;

use crate::errors::{DuplicateEiiImpls, EiiWithoutImpl};

/// Checks all EIIs in the crate graph, and returns for each declaration which implementation is
/// chosen. This could be a default implementation if no explicit implementation is found.
///
/// The returned map maps the defid of declaration macros to the defid of implementations.
pub(crate) fn get_externally_implementable_item_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    (): (),
) -> &'tcx FxIndexMap<LocalDefId, EiiMapping> {
    // We only need to check whether there are duplicate or missing EIIs if we're
    // emitting something that's not an rlib.
    let needs_check = tcx.crate_types().iter().any(|kind| match *kind {
        // Executables are leafs of the crate graph and need all EIIs to be satisfied,
        // either with defaults or explicit implementations. So they check their crate
        // graph to make sure this is the case.
        CrateType::Executable => true,
        // Proc macros are leafs of their crate graph and will be run,
        // and so need to check the EIIs of their dependencies.
        CrateType::ProcMacro => true,

        // These are a litte difficult. We don't know whether things depending on these
        // will perform checks to see if EIIs are implemented, or duplicated, or any other
        // of the checks performed in this function. So we must do the checks. However,
        // this can later lead to duplicate symbols when linking them together.
        // For this reason, we later mark EII symbols as "globally shared" and "may conflict".
        // In other words, if two shared libraries both provide an implementation for an EII,
        // that's fine! Just choose one... And because their mangled symbol names are the same
        // (that's exactly the conflict we're having) we hopefully have the same exact implementation.
        CrateType::Dylib | CrateType::Cdylib | CrateType::Staticlib | CrateType::Sdylib => true,

        // Rlibs are just a step in the crate graph.
        // Later on we'll link it together into an executable and over there we can check for EIIs
        CrateType::Rlib => false,
    });
    if !needs_check {
        // In this case we could only have called it when checking,
        // and not when we were actually codegenning functions so we don't need to return any real data
        return &*tcx.arena.alloc(FxIndexMap::default());
    }

    let mut eiis =
        FxIndexMap::<DefId, (EIIDecl, CrateNum, FxIndexMap<DefId, (EIIImpl, CrateNum)>)>::default();

    // println!("current crate: {}", tcx.crate_name(LOCAL_CRATE));

    // collect all the EII declarations, and possibly implementations from all descendent crates
    for &cnum in tcx.crates(()).iter().chain(iter::once(&LOCAL_CRATE)) {
        // println!("visiting crate: {}", tcx.crate_name(cnum));
        // get the eiis for the crate we're currently looking at
        let crate_eiis = tcx.externally_implementable_items(cnum);

        // update or insert the corresponding entries
        for (did, (decl, impls)) in crate_eiis {
            eiis.entry(*did)
                .or_insert_with(|| (*decl, cnum, Default::default()))
                .2
                .extend(impls.into_iter().map(|(did, i)| (*did, (*i, cnum))));
        }
    }

    let mut final_impls = FxIndexMap::default();

    // now we have all eiis! For each of them, choose one we want to actually generate.

    for (decl_did, (decl, decl_crate, impls)) in eiis {
        // println!("for decl: {decl_did:?}: {decl:?}");
        let mut default_impls = Vec::new();
        let mut explicit_impls = Vec::new();

        for (impl_did, (impl_metadata, cnum)) in impls {
            if impl_metadata.is_default {
                // println!("found default impl in {}", tcx.crate_name(cnum));
                default_impls.push((impl_did, cnum));
            } else {
                // println!("found impl in {}", tcx.crate_name(cnum));
                explicit_impls.push((impl_did, cnum));
            }
        }

        if explicit_impls.len() > 1 {
            tcx.dcx().emit_err(DuplicateEiiImpls {
                name: tcx.item_name(decl_did),
                first_span: tcx.def_span(explicit_impls[0].0),
                first_crate: tcx.crate_name(explicit_impls[0].1),
                second_span: tcx.def_span(explicit_impls[1].0),
                second_crate: tcx.crate_name(explicit_impls[1].1),

                help: (),

                additional_crates: (explicit_impls.len() > 2).then_some(()),
                num_additional_crates: explicit_impls.len() - 2,
                additional_crate_names: explicit_impls[2..]
                    .iter()
                    .map(|i| format!("`{}`", tcx.crate_name(i.1)))
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }

        if default_impls.len() > 1 {
            panic!("multiple not supported right now, but this is easily possible");
        }

        // println!("impls: {explicit_impls:?}");
        // println!("default impls: {default_impls:?}");

        if let Some((chosen_impl, _)) = explicit_impls.first().or(default_impls.first()) {
            let feed = tcx.create_def(
                CRATE_DEF_ID,
                Some(Symbol::intern(&format!("EII shim for {decl_did:?}"))),
                DefKind::Fn,
        None,
        &mut DisambiguatorState::new(),
            );

            let extern_item_did = decl.eii_extern_item;

            feed.generics_of(tcx.generics_of(extern_item_did).clone());
            feed.type_of(tcx.type_of(extern_item_did).clone());
            feed.def_span(tcx.def_span(chosen_impl));
            feed.visibility(tcx.visibility(chosen_impl));
            feed.feed_hir();

            // println!("generating {extern_item_did:?} for impl {chosen_impl:?} in crate {} with did {decl_did:?}", tcx.crate_name(LOCAL_CRATE));

            let shim_did = feed.def_id();

            // println!("shim: {shim_did:?}");

            final_impls.insert(
                shim_did,
                EiiMapping { extern_item: extern_item_did, chosen_impl: *chosen_impl },
            );
        } else {
            tcx.dcx().emit_err(EiiWithoutImpl {
                current_crate_name: tcx.crate_name(LOCAL_CRATE),
                decl_crate_name: tcx.crate_name(decl_crate),
                name: tcx.item_name(decl_did),
                span: decl.span,
                help: (),
            });
        }
    }

    tcx.arena.alloc(final_impls)
}
