#[legacy_exports];

#[legacy_exports]
#[merge = "check/mod.rs"]
pub mod check;
#[legacy_exports]
mod rscope;
#[legacy_exports]
mod astconv;
#[merge = "infer/mod.rs"]
mod infer;
#[legacy_exports]
mod collect;
#[legacy_exports]
mod coherence;
mod deriving;
