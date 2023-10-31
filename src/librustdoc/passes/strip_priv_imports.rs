//! Strips all private import statements (use, extern crate) from a
//! crate.
use crate::clean;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::{ImportStripper, Pass};

pub(crate) const STRIP_PRIV_IMPORTS: Pass = Pass {
    name: "strip-priv-imports",
    run: strip_priv_imports,
    description: "strips all private import statements (`use`, `extern crate`) from a crate",
};

pub(crate) fn strip_priv_imports(krate: clean::Crate, cx: &mut DocContext<'_>) -> clean::Crate {
    let is_json_output = cx.output_format.is_json() && !cx.show_coverage;
    ImportStripper {
        tcx: cx.tcx,
        is_json_output,
        document_hidden: cx.render_options.document_hidden,
    }
    .fold_crate(krate)
}
