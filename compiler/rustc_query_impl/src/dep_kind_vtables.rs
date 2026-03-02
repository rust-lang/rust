use rustc_middle::bug;
use rustc_middle::dep_graph::{DepKindVTable, DepNodeKey, KeyFingerprintStyle};
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
            is_anon: false,
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
            is_anon: false,
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
            is_anon: false,
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: Some(|tcx, _, prev_index| {
                tcx.dep_graph.force_diagnostic_node(tcx, prev_index);
                true
            }),
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn AnonZeroDeps<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: true,
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Opaque,
            force_from_dep_node_fn: Some(|_, _, _| bug!("cannot force an anon node")),
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn TraitSelect<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: true,
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Unit,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn CompileCodegenUnit<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Opaque,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn CompileMonoItem<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            key_fingerprint_style: KeyFingerprintStyle::Opaque,
            force_from_dep_node_fn: None,
            promote_from_disk_fn: None,
        }
    }

    pub(crate) fn Metadata<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
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
    is_anon: bool,
    is_cache_on_disk: bool,
    is_eval_always: bool,
) -> DepKindVTable<'tcx>
where
    Q: GetQueryVTable<'tcx>,
{
    let key_fingerprint_style = if is_anon {
        KeyFingerprintStyle::Opaque
    } else {
        <Q::Cache as QueryCache>::Key::key_fingerprint_style()
    };

    // A query dep-node can only be forced or promoted if it can recover a key
    // from its key fingerprint.
    let can_recover = key_fingerprint_style.is_maybe_recoverable();
    if is_anon {
        assert!(!can_recover);
    }

    DepKindVTable {
        is_anon,
        is_eval_always,
        key_fingerprint_style,
        force_from_dep_node_fn: can_recover.then_some(force_from_dep_node_inner::<Q>),
        promote_from_disk_fn: (can_recover && is_cache_on_disk)
            .then_some(promote_from_disk_inner::<Q>),
    }
}

/// Helper module containing a [`DepKindVTable`] constructor for each dep kind,
/// for use with [`rustc_middle::make_dep_kind_array`].
///
/// That macro will check that we gave it a constructor for every known dep kind.
mod _dep_kind_vtable_ctors {
    // Re-export all of the vtable constructors for non-query and query dep kinds.

    // Non-query vtable constructors are defined in normal code.
    pub(crate) use super::non_query::*;
    // Query vtable constructors are defined via a macro.
    pub(crate) use crate::_dep_kind_vtable_ctors_for_queries::*;
}

pub fn make_dep_kind_vtables<'tcx>(
    arena: &'tcx rustc_middle::arena::Arena<'tcx>,
) -> &'tcx [DepKindVTable<'tcx>] {
    // Create an array of vtables, one for each dep kind (non-query and query).
    let dep_kind_vtables: [DepKindVTable<'tcx>; _] =
        rustc_middle::make_dep_kind_array!(_dep_kind_vtable_ctors);
    arena.alloc_from_iter(dep_kind_vtables)
}
