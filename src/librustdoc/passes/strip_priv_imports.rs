//! Strips all private import statements (use, extern crate) from a
//! crate.
use crate::clean;
use crate::core::DocContext;
use crate::fold::DocFolder;
use crate::passes::{ImportStripper, Pass};

crate const STRIP_PRIV_IMPORTS: Pass = Pass {
    name: "strip-priv-imports",
    run: strip_priv_imports,
    description: "strips all private import statements (`use`, `extern crate`) from a crate",
};

crate fn strip_priv_imports(krate: clean::Crate, _: &mut DocContext<'_>) -> clean::Crate {
    ImportStripper.fold_crate(krate)
}
