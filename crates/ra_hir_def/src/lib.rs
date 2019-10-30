use ra_arena::{impl_arena_id, RawId};
use ra_db::CrateId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId {
    pub krate: CrateId,
    pub module_id: CrateModuleId,
}

/// An ID of a module, **local** to a specific crate
// FIXME: rename to `LocalModuleId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CrateModuleId(RawId);
impl_arena_id!(CrateModuleId);
