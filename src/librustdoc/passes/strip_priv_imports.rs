use crate::clean;
use crate::fold::{DocFolder};
use crate::core::DocContext;
use crate::passes::{ImportStripper, Pass};

pub const STRIP_PRIV_IMPORTS: Pass = Pass::early("strip-priv-imports", strip_priv_imports,
     "strips all private import statements (`use`, `extern crate`) from a crate");

pub fn strip_priv_imports(krate: clean::Crate, _: &DocContext<'_, '_, '_>)  -> clean::Crate {
    ImportStripper.fold_crate(krate)
}
