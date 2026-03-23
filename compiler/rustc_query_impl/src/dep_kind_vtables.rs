use rustc_middle::bug;
use rustc_middle::dep_graph::{DepKind, DepKindVTable, DepNodeKey, KeyFingerprintStyle};
use rustc_middle::query::QueryCache;

use crate::GetQueryVTable;
use crate::plumbing::{force_from_dep_node_inner, promote_from_disk_inner};

/// [`DepKindVTable`] constructors for special dep kinds that aren't queries.
#[expect(non_snake_case, reason = "use non-snake case to avoid collision with query names")]
mod non_query {
    use super::*;

    // We use this for most things when incr. comp. is turned off.
    pub(crate) fn Null<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: Some(|_, dep_node, _| {
                bug!("force_from_dep_node: encountered {dep_node:?}")
            }),
            promote_from_disk_fn: None,
        }
    }

    // We use this for the forever-red node.
    pub(crate) fn Red<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: Some(|_, dep_node, _| {
                bug!("force_from_dep_node: encountered {dep_node:?}")
            }),
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn SideEffect<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: Some(|tcx, _, prev_index| {
                tcx.dep_graph.force_side_effect(tcx, prev_index);
                true
            }),
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn AnonZeroDeps<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Opaque,
            force_from_dep_node_fn: Some(|_, _, _| bug!("cannot force an anon node")),
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn TraitSelect<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn CompileCodegenUnit<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Opaque,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn CompileMonoItem<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Opaque,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn Metadata<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }
}

/// Shared implementation of the [`DepKindVTable`] constructor for queries.
/// Called from macro-generated code for each query.
pub(crate) fn make_dep_kind_vtable_for_query<'tcx, Q>(
    is_cache_on_disk: bool,
    is_eval_always: bool,
    is_no_force: bool,
) -> DepKindVTable<'tcx>
where
    Q: GetQueryVTable<'tcx>,
{
    // A query dep-node can only be forced or promoted if it can recover a key
    // from its key fingerprint.
    let key_fingerprint_style = <Q::Cache as QueryCache>::Key::key_fingerprint_style();
    let can_recover = key_fingerprint_style.is_maybe_recoverable();

    DepKindVTable {
        is_eval_always,
        key_fingerprint_style,
        force_from_dep_node_fn: (can_recover && !is_no_force)
            .then_some(force_from_dep_node_inner::<Q>),
        promote_from_disk_fn: (can_recover && is_cache_on_disk)
            .then_some(promote_from_disk_inner::<Q>),
    }
}

macro_rules! define_dep_kind_vtables {
    (
        queries {
            $(
                $(#[$attr:meta])*
                fn $name:ident($K:ty) -> $V:ty
                {
                    // Search for (QMODLIST) to find all occurrences of this query modifier list.
                    arena_cache: $arena_cache:literal,
                    cache_on_disk: $cache_on_disk:literal,
                    depth_limit: $depth_limit:literal,
                    eval_always: $eval_always:literal,
                    feedable: $feedable:literal,
                    no_force: $no_force:literal,
                    no_hash: $no_hash:literal,
                    returns_error_guaranteed: $returns_error_guaranteed:literal,
                    separate_provide_extern: $separate_provide_extern:literal,
                }
            )*
        }
        non_queries {
            $(
                $(#[$nq_attr:meta])*
                $nq_name:ident,
            )*
        }
    ) => {{
        [
            // The small number of non-query vtables: `Null`, `Red`, etc.
            $(
                non_query::$nq_name(),
            )*

            // The large number of query vtables.
            $(
                $crate::dep_kind_vtables::make_dep_kind_vtable_for_query::<
                    $crate::query_impl::$name::VTableGetter,
                >(
                    $cache_on_disk,
                    $eval_always,
                    $no_force,
                )
            ),*
        ]
    }}
}

// Create an array of vtables, one for each dep kind (non-query and query).
pub(crate) fn make_dep_kind_vtables<'tcx>() -> [DepKindVTable<'tcx>; DepKind::NUM_VARIANTS] {
    rustc_middle::queries::rustc_with_all_queries! { define_dep_kind_vtables! }
}
