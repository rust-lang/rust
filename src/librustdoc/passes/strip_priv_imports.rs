//! Strips all private import statements (use, extern crate) from a
//! crate.

use std::alloc::Allocator;

use crate::clean;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::{ImportStripper, Pass};

pub(crate) fn strip_priv_imports_pass<A: Allocator + Copy>() -> Pass<A> {
    Pass {
        name: "strip-priv-imports",
        run: Some(strip_priv_imports),
        description: "strips all private import statements (`use`, `extern crate`) from a crate",
    }
}

pub(crate) fn strip_priv_imports<A: Allocator + Copy>(
    krate: clean::Crate,
    cx: &mut DocContext<'_, A>,
) -> clean::Crate {
    let is_json_output = cx.is_json_output();
    ImportStripper { tcx: cx.tcx, is_json_output, document_hidden: cx.document_hidden() }
        .fold_crate(krate, cx.cache.search_index.allocator())
}
