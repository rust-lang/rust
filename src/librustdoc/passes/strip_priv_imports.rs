use clean;
use core::DocContext;
use fold::DocFolder;
use passes::{ImportStripper, Pass};

pub const STRIP_PRIV_IMPORTS: Pass = Pass::early("strip-priv-imports", strip_priv_imports,
     "strips all private import statements (`use`, `extern crate`) from a crate");

pub fn strip_priv_imports(krate: clean::Crate, _: &DocContext)  -> clean::Crate {
    ImportStripper.fold_crate(krate)
}
