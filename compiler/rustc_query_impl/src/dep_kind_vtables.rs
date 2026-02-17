use rustc_middle::bug;
use rustc_middle::dep_graph::{DepKindVTable, DepNodeKey, FingerprintStyle};
use rustc_middle::query::QueryCache;

use crate::plumbing::{force_from_dep_node_inner, try_load_from_on_disk_cache_inner};
use crate::{QueryDispatcherUnerased, QueryFlags};

/// [`DepKindVTable`] constructors for special dep kinds that aren't queries.
#[expect(non_snake_case, reason = "use non-snake case to avoid collision with query names")]
mod non_query {
    use super::*;

    // We use this for most things when incr. comp. is turned off.
    pub(crate) fn Null<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Unit,
            force_from_dep_node: Some(|_, dep_node, _| {
                bug!("force_from_dep_node: encountered {dep_node:?}")
            }),
            try_load_from_on_disk_cache: None,
            name: &"Null",
        }
    }

    // We use this for the forever-red node.
    pub(crate) fn Red<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Unit,
            force_from_dep_node: Some(|_, dep_node, _| {
                bug!("force_from_dep_node: encountered {dep_node:?}")
            }),
            try_load_from_on_disk_cache: None,
            name: &"Red",
        }
    }

    pub(crate) fn SideEffect<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Unit,
            force_from_dep_node: Some(|tcx, _, prev_index| {
                tcx.dep_graph.force_diagnostic_node(tcx, prev_index);
                true
            }),
            try_load_from_on_disk_cache: None,
            name: &"SideEffect",
        }
    }

    pub(crate) fn AnonZeroDeps<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: true,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Opaque,
            force_from_dep_node: Some(|_, _, _| bug!("cannot force an anon node")),
            try_load_from_on_disk_cache: None,
            name: &"AnonZeroDeps",
        }
    }

    pub(crate) fn TraitSelect<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: true,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Unit,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: &"TraitSelect",
        }
    }

    pub(crate) fn CompileCodegenUnit<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Opaque,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: &"CompileCodegenUnit",
        }
    }

    pub(crate) fn CompileMonoItem<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Opaque,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: &"CompileMonoItem",
        }
    }

    pub(crate) fn Metadata<'tcx>() -> DepKindVTable<'tcx> {
        DepKindVTable {
            is_anon: false,
            is_eval_always: false,
            fingerprint_style: FingerprintStyle::Unit,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: &"Metadata",
        }
    }
}

/// Shared implementation of the [`DepKindVTable`] constructor for queries.
/// Called from macro-generated code for each query.
pub(crate) fn make_dep_kind_vtable_for_query<'tcx, Q, Cache, const FLAGS: QueryFlags>(
    is_eval_always: bool,
) -> DepKindVTable<'tcx>
where
    Q: QueryDispatcherUnerased<'tcx, Cache, FLAGS>,
    Cache: QueryCache + 'tcx,
{
    let is_anon = FLAGS.is_anon;
    let fingerprint_style = if is_anon {
        FingerprintStyle::Opaque
    } else {
        <Cache::Key as DepNodeKey<'tcx>>::fingerprint_style()
    };

    if is_anon || !fingerprint_style.reconstructible() {
        return DepKindVTable {
            is_anon,
            is_eval_always,
            fingerprint_style,
            force_from_dep_node: None,
            try_load_from_on_disk_cache: None,
            name: Q::NAME,
        };
    }

    DepKindVTable {
        is_anon,
        is_eval_always,
        fingerprint_style,
        force_from_dep_node: Some(|tcx, dep_node, _| {
            force_from_dep_node_inner(Q::query_dispatcher(tcx), tcx, dep_node)
        }),
        try_load_from_on_disk_cache: Some(|tcx, dep_node| {
            try_load_from_on_disk_cache_inner(Q::query_dispatcher(tcx), tcx, dep_node)
        }),
        name: Q::NAME,
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
