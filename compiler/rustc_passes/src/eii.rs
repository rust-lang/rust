//! Validity checking for weak lang items

use std::iter;

use rustc_attr_parsing::{EIIDecl, EIIImpl};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CRATE_DEF_ID, CrateNum, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::definitions::DisambiguatorState;
use rustc_middle::bug;
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
    #[derive(Copy, Clone)]
    enum Case {
        /// We need to generate all EII shims because we are generating some final target like an
        /// executable or library (not rlib)
        AlwaysEmit,
        /// We need to generate all EII shims because one of our crate types is a final target like
        /// an executable. However, we're also generating an rlib. So. If we see explicit
        /// definitions of EIIs we can generate them with external linkage. However, if we find
        /// defaults, they must also be emitted because some of our crate types are final targets.
        /// And unfortunately the rlib will also contain these definitions. However, because rlibs
        /// will later be used in final targets, which will use `AlwaysEmit`, these symbols that were
        /// spuriously generated in rlibs will be redefined and then flagged by the linker as
        /// duplicate definitions. So, we have to emit EII shims which are default impls (not
        /// explicit ones) as weak symbols.
        EmitMaybeWeak,
        /// We don't always need to emit EIIs because we're generating an Rlib. However, if we see
        /// an explicit implementation, we can! Because it cannot be overwritten anymore.
        EmitExternalIfExplicit,
    }

    let has_rlib = tcx.crate_types().iter().any(|i| matches!(i, CrateType::Rlib));
    let has_target = tcx.crate_types().iter().any(|i| !matches!(i, CrateType::Rlib));

    let case = match (has_rlib, has_target) {
        (true, true) => Case::EmitMaybeWeak,
        (true, false) => Case::EmitExternalIfExplicit,
        (false, true) => Case::AlwaysEmit,
        (false, false) => {
            bug!("no targets but somehow we are running the compiler")
        }
    };

    #[derive(Debug)]
    struct FoundImpl {
        imp: EIIImpl,
        impl_crate: CrateNum,
    }

    #[derive(Debug)]
    struct FoundEii {
        decl: EIIDecl,
        decl_crate: CrateNum,
        impls: FxIndexMap<DefId, FoundImpl>,
    }

    let mut eiis = FxIndexMap::<DefId, FoundEii>::default();

    // println!("current crate: {}", tcx.crate_name(LOCAL_CRATE));

    // collect all the EII declarations, and possibly implementations from all descendent crates
    for &cnum in tcx.crates(()).iter().chain(iter::once(&LOCAL_CRATE)) {
        // println!("visiting crate: {}", tcx.crate_name(cnum));
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

    let mut final_impls = FxIndexMap::default();

    // now we have all eiis! For each of them, choose one we want to actually generate.

    for (decl_did, FoundEii { decl, decl_crate, impls }) in eiis {
        // println!("for decl: {decl_did:?}: {decl:?}");
        let mut default_impls = Vec::new();
        let mut explicit_impls = Vec::new();

        for (impl_did, FoundImpl { imp, impl_crate }) in impls {
            if imp.is_default {
                // println!("found default impl in {}", tcx.crate_name(cnum));
                default_impls.push((impl_did, impl_crate));
            } else {
                // println!("found impl in {}", tcx.crate_name(cnum));
                explicit_impls.push((impl_did, impl_crate));
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

        let (chosen_impl, weak_linkage) =
            match (case, explicit_impls.first(), default_impls.first()) {
                (Case::EmitExternalIfExplicit, Some((explicit, impl_crate)), _) => {
                    if impl_crate != &LOCAL_CRATE {
                        continue;
                    }
                    (explicit, false)
                }
                // we don't care in this case if we find no implementation yet. Another can come
                // downstream.
                (Case::EmitExternalIfExplicit, None, _) => {
                    continue;
                }
                (Case::AlwaysEmit, Some((explicit, impl_crate)), _) => {
                    if impl_crate != &LOCAL_CRATE {
                        continue;
                    }

                    (explicit, false)
                }
                (Case::AlwaysEmit, _, Some((deflt, _))) => (deflt, false),

                (Case::EmitMaybeWeak, Some((explicit, impl_crate)), _) => {
                    if impl_crate != &LOCAL_CRATE {
                        continue;
                    }

                    (explicit, false)
                }
                // IMPORTANT! weak linkage because the symbol will also end up in the rlib and may need
                // to be overwritten :(
                (Case::EmitMaybeWeak, _, Some((deflt, _))) => (deflt, true),

                // We have a target to generate, but no impl to put in it. error!
                (Case::EmitMaybeWeak | Case::AlwaysEmit, None, None) => {
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
            EiiMapping { extern_item: extern_item_did, chosen_impl: *chosen_impl, weak_linkage },
        );
    }

    tcx.arena.alloc(final_impls)
}
