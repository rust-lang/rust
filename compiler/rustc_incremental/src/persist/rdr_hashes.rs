use rustc_hir::attrs::RDRFields;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::find_attr;
use rustc_macros::Diagnostic;
use rustc_middle::dep_graph::{DepKind, DepNode};
use rustc_middle::ty::TyCtxt;
use rustc_span::{Span, Symbol};
use tracing::debug;

pub(crate) fn check_rdr_test_attrs(tcx: TyCtxt<'_>) {
    // can't add the attributes without opting into this feature
    if !tcx.features().rustc_attrs() {
        return;
    }
    for &(span, fields) in
        find_attr!(tcx.hir_attrs(rustc_hir::CRATE_HIR_ID), RustcRDRTestAttr(e) => e)
            .into_iter()
            .flatten()
    {
        assert_dependency_public_hash(tcx, span, fields);
    }
}

#[derive(Diagnostic)]
#[diag("found rdr hash attribute but `-Zquery-dep-graph` was not specified")]
pub(crate) struct MissingQueryDepGraph {
    #[primary_span]
    pub span: Span,
}

/// Scan for a `cfg="foo"` attribute and check whether we have a
/// cfg flag called `foo`.
fn check_config(tcx: TyCtxt<'_>, value: Symbol) -> bool {
    let config = &tcx.sess.config;
    debug!("check_config(config={:?}, value={:?})", config, value);
    if config.iter().any(|&(name, _)| name == value) {
        debug!("check_config: matched");
        return true;
    }
    debug!("check_config: no match found");
    false
}

fn assert_dependency_public_hash(tcx: TyCtxt<'_>, span: Span, fields: RDRFields) {
    if !tcx.sess.opts.unstable_opts.query_dep_graph {
        tcx.dcx().emit_fatal(MissingQueryDepGraph { span });
    }

    let crate_num = tcx
        .crates(())
        .iter()
        .copied()
        .find(|&cnum| tcx.crate_name(cnum).as_str() == fields.crate_name.as_str())
        .unwrap_or_else(|| {
            tcx.dcx().span_fatal(
                span,
                format!("crate `{}` not found in dependencies", fields.crate_name),
            )
        });
    if crate_num == LOCAL_CRATE {
        tcx.dcx().span_fatal(span, "expected the name of a dependency crate");
    }

    if !check_config(tcx, fields.cfg) {
        debug!("check_attr: config does not match, ignoring attr");
        return;
    }

    let green = !fields.changed;
    let dep_node = DepNode::construct(tcx, DepKind::public_api_hash, &crate_num);
    let is_green = tcx.dep_graph.is_green(&dep_node);
    let is_red = tcx.dep_graph.is_red(&dep_node);
    if !is_red && !is_green {
        tcx.dcx().span_fatal(span, "dependency color is neither red or green!");
    }

    if green && !is_green {
        tcx.dcx().span_fatal(
            span,
            "expected dependency to be unchanged (green) but it was changed (red)",
        );
    } else if !green && is_green {
        tcx.dcx().span_fatal(
            span,
            "expected dependency to have changed (red) but it was unchanged (green)",
        );
    }
}
