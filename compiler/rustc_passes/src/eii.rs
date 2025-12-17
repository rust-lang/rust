//! Checks necessary for externally implementable items:
//! Are all items implemented etc.?

use std::iter;

use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::attrs::{EiiDecl, EiiImpl};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::CrateType;

use crate::errors::{DuplicateEiiImpls, EiiWithoutImpl};

#[derive(Clone, Copy, Debug)]
enum CheckingMode {
    CheckDuplicates,
    CheckExistence,
}

fn get_checking_mode(tcx: TyCtxt<'_>) -> CheckingMode {
    // if any of the crate types is not rlib or dylib, we must check for existence.
    if tcx.crate_types().iter().any(|i| !matches!(i, CrateType::Rlib | CrateType::Dylib)) {
        CheckingMode::CheckExistence
    } else {
        CheckingMode::CheckDuplicates
    }
}

/// Checks for a given crate, what EIIs need to be generated in it.
/// This is usually a small subset of all EIIs.
///
/// EII implementations come in two varieties: explicit and default.
/// This query is called once for every crate, to check whether there aren't any duplicate explicit implementations.
/// A duplicate may be caused by an implementation in the current crate,
/// though it's also entirely possible that the source is two dependencies with an explicit implementation.
/// Those work fine on their own but the combination of the two is a conflict.
///
/// However, if the current crate is a "root" crate, one that generates a final artifact like a binary,
/// then we check one more thing, namely that every EII actually has an implementation, either default or not.
/// If one EII has no implementation, that's an error at that point.
///
/// These two behaviors are implemented using `CheckingMode`.
pub(crate) fn check_externally_implementable_items<'tcx>(tcx: TyCtxt<'tcx>, (): ()) {
    let checking_mode = get_checking_mode(tcx);

    #[derive(Debug)]
    struct FoundImpl {
        imp: EiiImpl,
        impl_crate: CrateNum,
    }

    #[derive(Debug)]
    struct FoundEii {
        decl: EiiDecl,
        decl_crate: CrateNum,
        impls: FxIndexMap<DefId, FoundImpl>,
    }

    let mut eiis = FxIndexMap::<DefId, FoundEii>::default();

    // collect all the EII declarations, and possibly implementations from all descendent crates
    for &cnum in tcx.crates(()).iter().chain(iter::once(&LOCAL_CRATE)) {
        // get the eiis for the crate we're currently looking at
        let crate_eiis = tcx.externally_implementable_items(cnum);

        // update or insert the corresponding entries
        for (did, (decl, impls)) in crate_eiis {
            eiis.entry(*did)
                .or_insert_with(|| FoundEii {
                    decl: *decl,
                    decl_crate: cnum,
                    impls: Default::default(),
                })
                .impls
                .extend(
                    impls
                        .into_iter()
                        .map(|(did, i)| (*did, FoundImpl { imp: *i, impl_crate: cnum })),
                );
        }
    }

    // now we have all eiis! For each of them, choose one we want to actually generate.
    for (decl_did, FoundEii { decl, decl_crate, impls }) in eiis {
        let mut default_impls = Vec::new();
        let mut explicit_impls = Vec::new();

        for (impl_did, FoundImpl { imp, impl_crate }) in impls {
            if imp.is_default {
                default_impls.push((impl_did, impl_crate));
            } else {
                explicit_impls.push((impl_did, impl_crate));
            }
        }

        // more than one explicit implementation (across all crates)
        // is instantly an error.
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
            let decl_span = tcx.def_ident_span(decl_did).unwrap();
            tcx.dcx().span_delayed_bug(decl_span, "multiple not supported right now");
        }

        let (local_impl, is_default) =
            // note, for a single crate we never need to generate both a default and an explicit implementation.
            // In that case, generating the explicit implementation is enough!
            match (checking_mode, explicit_impls.first(), default_impls.first()) {
                // If we find an explicit implementation, it's instantly the chosen implementation.
                (_, Some((explicit, _)), _) => (explicit, false),
                // if we find a default implementation, we can emit it but the alias should be weak
                (_, _, Some((deflt, _))) => (deflt, true),

                // if we find no explicit implementation,
                // that's fine if we're only checking for duplicates.
                // The existence will be checked somewhere else in a crate downstream.
                (CheckingMode::CheckDuplicates, None, _) => continue,

                // We have a target to generate, but no impl to put in it. error!
                (CheckingMode::CheckExistence, None, None) => {
                    tcx.dcx().emit_err(EiiWithoutImpl {
                        current_crate_name: tcx.crate_name(LOCAL_CRATE),
                        decl_crate_name: tcx.crate_name(decl_crate),
                        name: tcx.item_name(decl_did),
                        span: decl.span,
                        help: (),
                    });

                    continue;
                }
            };

        // if it's not local, who cares about generating it.
        // That's the local crates' responsibility
        let Some(chosen_impl) = local_impl.as_local() else {
            continue;
        };

        tracing::debug!("generating EII {chosen_impl:?} (default={is_default})");
    }
}
