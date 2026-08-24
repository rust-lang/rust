//! The definitions of *passes* which transform crate information.

use crate::clean::Crate;
use crate::core::DocContext;

mod stripper;
pub(crate) use stripper::*;

pub(crate) mod check_doc_test_visibility;
pub(crate) mod collect_intra_doc_links;
mod collect_trait_impls;
mod lint;
mod propagate_doc_cfg;
mod propagate_stability;
mod strip_aliased_non_local;
mod strip_hidden;
mod strip_priv_imports;
mod strip_private;

#[derive(Default)]
pub(crate) struct Store {
    links: collect_intra_doc_links::LinkCollection,
}

#[tracing::instrument(level = "info", skip_all)]
pub(crate) fn run(
    mut krate: Crate,
    cx: &mut DocContext<'_>,
    show_coverage: bool,
) -> (Crate, Store) {
    macro_rules! run {
        ($name:ident($( $args:tt )*)) => {{
            tracing::debug!("running pass `{}`", stringify!($name));
            cx.tcx.sess.time(stringify!($name), || $name::$name($( $args )*))
        }};
    }

    let mut store = Store::default();

    if !show_coverage {
        krate = run!(collect_trait_impls(krate, cx));
        krate = run!(check_doc_test_visibility(krate, cx));
        krate = run!(strip_aliased_non_local(krate, cx));
        krate = run!(propagate_doc_cfg(krate, cx));
    }

    krate = run!(strip_hidden(krate, cx));
    krate = run!(strip_private(krate, cx));

    if !show_coverage {
        krate = run!(strip_priv_imports(krate, cx));
        (krate, store.links) = run!(collect_intra_doc_links(krate, cx));
        krate = run!(propagate_stability(krate, cx));
        krate = run!(lint(krate, cx));
    }

    (krate, store)
}

/// To be run after the cache in [`DocContext`] has been fully populated.
pub(crate) fn finalize(cx: &mut DocContext<'_>, store: Store) {
    collect_intra_doc_links::resolve_ambiguous_links(store.links, cx);
}
