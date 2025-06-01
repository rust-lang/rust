use rustc_hir::def_id::LocalDefId;
use rustc_macros::HashStable;

#[derive(Debug, HashStable)]
pub enum DistributedSliceAddition {
    Single(LocalDefId),
    Many(LocalDefId),
}
