//! Throughout the compiler tree, there are several places which want to have
//! access to state or queries while being inside crates that are dependencies
//! of `rustc_middle`. To facilitate this, we have the
//! `rustc_data_structures::AtomicRef` type, which allows us to setup a global
//! static which can then be set in this file at program startup.
//!
//! See `SPAN_TRACK` for an example of how to set things up.
//!
//! The functions in this file should fall back to the default set in their
//! origin crate when the `TyCtxt` is not present in TLS.

use std::fmt;

use rustc_errors::{DiagInner, TRACK_DIAGNOSTIC};
use rustc_middle::dep_graph::{DepNodeExt, TaskDepsRef};
use rustc_middle::ty::tls;
use rustc_query_impl::QueryCtxt;
use rustc_query_system::dep_graph::dep_node::default_dep_kind_debug;
use rustc_query_system::dep_graph::{DepContext, DepKind, DepNode};

fn track_span_parent(def_id: rustc_span::def_id::LocalDefId) {
    tls::with_context_opt(|icx| {
        if let Some(icx) = icx {
            // `track_span_parent` gets called a lot from HIR lowering code.
            // Skip doing anything if we aren't tracking dependencies.
            let tracks_deps = match icx.task_deps {
                TaskDepsRef::Allow(..) => true,
                TaskDepsRef::EvalAlways | TaskDepsRef::Ignore | TaskDepsRef::Forbid => false,
            };
            if tracks_deps {
                let _span = icx.tcx.source_span(def_id);
                // Sanity check: relative span's parent must be an absolute span.
                debug_assert_eq!(_span.data_untracked().parent, None);
            }
        }
    })
}

/// This is a callback from `rustc_errors` as it cannot access the implicit state
/// in `rustc_middle` otherwise. It is used when diagnostic messages are
/// emitted and stores them in the current query, if there is one.
fn track_diagnostic<R>(diagnostic: DiagInner, f: &mut dyn FnMut(DiagInner) -> R) -> R {
    tls::with_context_opt(|icx| {
        if let Some(icx) = icx {
            icx.tcx.dep_graph.record_diagnostic(QueryCtxt::new(icx.tcx), &diagnostic);

            // Diagnostics are tracked, we can ignore the dependency.
            let icx = tls::ImplicitCtxt { task_deps: TaskDepsRef::Ignore, ..icx.clone() };
            tls::enter_context(&icx, move || (*f)(diagnostic))
        } else {
            // In any other case, invoke diagnostics anyway.
            (*f)(diagnostic)
        }
    })
}

/// This is a callback from `rustc_hir` as it cannot access the implicit state
/// in `rustc_middle` otherwise.
fn def_id_debug(def_id: rustc_hir::def_id::DefId, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "DefId({}:{}", def_id.krate, def_id.index.index())?;
    tls::with_opt(|opt_tcx| {
        if let Some(tcx) = opt_tcx {
            write!(f, " ~ {}", tcx.def_path_debug_str(def_id))?;
        }
        Ok(())
    })?;
    write!(f, ")")
}

/// This is a callback from `rustc_query_system` as it cannot access the implicit state
/// in `rustc_middle` otherwise.
pub fn dep_kind_debug(kind: DepKind, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    tls::with_opt(|opt_tcx| {
        if let Some(tcx) = opt_tcx {
            write!(f, "{}", tcx.dep_kind_info(kind).name)
        } else {
            default_dep_kind_debug(kind, f)
        }
    })
}

/// This is a callback from `rustc_query_system` as it cannot access the implicit state
/// in `rustc_middle` otherwise.
pub fn dep_node_debug(node: DepNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}(", node.kind)?;

    tls::with_opt(|opt_tcx| {
        if let Some(tcx) = opt_tcx {
            if let Some(def_id) = node.extract_def_id(tcx) {
                write!(f, "{}", tcx.def_path_debug_str(def_id))?;
            } else if let Some(ref s) = tcx.dep_graph.dep_node_debug_str(node) {
                write!(f, "{s}")?;
            } else {
                write!(f, "{}", node.hash)?;
            }
        } else {
            write!(f, "{}", node.hash)?;
        }
        Ok(())
    })?;

    write!(f, ")")
}

/// Sets up the callbacks in prior crates which we want to refer to the
/// TyCtxt in.
pub fn setup_callbacks() {
    rustc_span::SPAN_TRACK.swap(&(track_span_parent as fn(_)));
    rustc_hir::def_id::DEF_ID_DEBUG.swap(&(def_id_debug as fn(_, &mut fmt::Formatter<'_>) -> _));
    rustc_query_system::dep_graph::dep_node::DEP_KIND_DEBUG
        .swap(&(dep_kind_debug as fn(_, &mut fmt::Formatter<'_>) -> _));
    rustc_query_system::dep_graph::dep_node::DEP_NODE_DEBUG
        .swap(&(dep_node_debug as fn(_, &mut fmt::Formatter<'_>) -> _));
    TRACK_DIAGNOSTIC.swap(&(track_diagnostic as _));
}
