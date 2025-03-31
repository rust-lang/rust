//! Validity checking for weak lang items

use std::iter;

use rustc_attr_parsing::EIIImpl;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CRATE_DEF_ID, CrateNum, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::definitions::DisambiguatorState;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::CrateType;
use rustc_span::Symbol;

use crate::errors::{DuplicateEiiImpls, EiiWithoutImpl};

/// Checks all EIIs in the crate graph, and returns for each declaration which implementation is
/// chosen. This could be a default implementation if no explicit implementation is found.
///
/// The returned map maps the defid of declaration macros to the defid of implementations.
pub(crate) fn get_eii_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    (): (),
) -> &'tcx FxIndexMap<DefId, (DefId, LocalDefId)> {
    // We only need to check whether there are duplicate or missing EIIs if we're
    // emitting something that's not an rlib.
    let needs_check = tcx.crate_types().iter().any(|kind| match *kind {
        CrateType::Dylib
            | CrateType::ProcMacro
            | CrateType::Cdylib
            | CrateType::Executable
            | CrateType::Staticlib
            | CrateType::Sdylib => true,
        CrateType::Rlib => false,
    });
    if !needs_check {
        // In this case we could only have called it when checking,
        // and not when we were actually codegenning functions so we don't need to return any real data
        return &*tcx.arena.alloc(FxIndexMap::default());
    }

    let mut eiis = FxIndexMap::<_, (_, FxIndexMap<DefId, (EIIImpl, CrateNum)>)>::default();

    // println!("current crate: {}", tcx.crate_name(LOCAL_CRATE));

    // collect all the EII declarations, and possibly implementations from all descendent crates
    for &cnum in tcx.crates(()).iter().chain(iter::once(&LOCAL_CRATE)) {
        // println!("visiting crate: {}", tcx.crate_name(cnum));
        // get the eiis for the crate we're currently looking at
        let crate_eiis = tcx.externally_implementable_items(cnum);

        // update or insert the corresponding entries
        for (did, (decl, impls)) in crate_eiis {
            eiis.entry(did)
                .or_insert_with(|| (decl, Default::default()))
                .1
                .extend(impls.into_iter().map(|(did, i)| (*did, (*i, cnum))));
        }
    }

    let mut final_impls = FxIndexMap::default();

    // now we have all eiis! For each of them, choose one we want to actually generate.

    for (decl_did, (decl, impls)) in eiis {
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
                name: tcx.item_name(*decl_did),
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
            feed.generics_of(tcx.generics_of(*chosen_impl).clone());
            feed.type_of(tcx.type_of(*chosen_impl).clone());
            feed.def_span(tcx.def_span(*chosen_impl));
            feed.feed_hir();

            let shim_did = feed.def_id();

            // println!("shim: {shim_did:?}");

            final_impls.insert(*decl_did, (*chosen_impl, shim_did));
        } else {
            tcx.dcx().emit_err(EiiWithoutImpl {
                current_crate_name: tcx.crate_name(LOCAL_CRATE),
                name: tcx.item_name(*decl_did),
                span: decl.span,
                help: (),
            });
        }
    }

    tcx.arena.alloc(final_impls)
}
