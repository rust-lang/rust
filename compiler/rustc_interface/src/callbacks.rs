//! Throughout the compiler tree, there are several places which want to have
//! access to state or queries while being inside crates that are dependencies
//! of `rustc_middle`. To facilitate this, we have the
//! `rustc_data_structures::AtomicRef` type, which allows us to setup a global
//! static which can then be set in this file at program startup.
//!
//! See `SPAN_DEBUG` for an example of how to set things up.
//!
//! The functions in this file should fall back to the default set in their
//! origin crate when the `TyCtxt` is not present in TLS.

use rustc_middle::dep_graph::DepNodeExt;
use rustc_middle::ty::tls;
use rustc_query_system::dep_graph::DepNode;
use std::fmt;

/// This is a callback from `rustc_ast` as it cannot access the implicit state
/// in `rustc_middle` otherwise.
fn span_debug(span: rustc_span::Span, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    tls::with_opt(|tcx| {
        if let Some(tcx) = tcx {
            rustc_span::debug_with_source_map(span, f, tcx.sess.source_map())
        } else {
            rustc_span::default_span_debug(span, f)
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

fn dep_node_debug(node: &DepNode, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", node.kind)?;

    if !node.kind.has_params && !node.kind.is_anon {
        return Ok(());
    }

    write!(f, "(")?;

    tls::with_opt(|opt_tcx| {
        if let Some(tcx) = opt_tcx {
            if let Some(def_id) = node.extract_def_id(tcx) {
                write!(f, "{}", tcx.def_path_debug_str(def_id))?;
            } else if let Some(ref s) = tcx.dep_graph.dep_node_debug_str(*node) {
                write!(f, "{}", s)?;
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
    rustc_span::SPAN_DEBUG.swap(&(span_debug as fn(_, &mut fmt::Formatter<'_>) -> _));
    rustc_hir::def_id::DEF_ID_DEBUG.swap(&(def_id_debug as fn(_, &mut fmt::Formatter<'_>) -> _));
    rustc_query_system::dep_graph::NODE_DEBUG.swap(&(dep_node_debug as _));
    rustc_errors::TRACK_DIAGNOSTICS.swap(&(rustc_query_system::tls::track_diagnostic as fn(&_)));
}
