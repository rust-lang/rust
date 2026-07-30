//! The definitions of *passes* which transform crate information.

mod stripper;
pub(crate) use stripper::*;

pub(crate) mod calculate_doc_coverage;
pub(crate) mod check_doc_test_visibility;
pub(crate) mod collect_intra_doc_links;
pub(crate) mod collect_trait_impls;
pub(crate) mod lint;
pub(crate) mod propagate_doc_cfg;
pub(crate) mod propagate_stability;
pub(crate) mod strip_aliased_non_local;
pub(crate) mod strip_hidden;
pub(crate) mod strip_priv_imports;
pub(crate) mod strip_private;

macro_rules! initialize {
    ($tcx:ident, $cx:ident) => {
        macro_rules! track {
            ($name:ident($krate:ident)) => {{
                debug!("running pass `{}`", stringify!($name));
                $tcx.sess.time(stringify!($name), || passes::$name::$name($krate, &mut $cx))
            }};
        }
    };
}

pub(crate) use initialize;
