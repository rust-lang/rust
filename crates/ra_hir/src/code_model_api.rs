use ra_db::{CrateId, Cancelable};

use crate::{Module, Name, db::HirDatabase};

/// hir::Crate describes a single crate. It's the main inteface with which
/// crate's dependencies interact. Mostly, it should be just a proxy for the
/// root module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Crate {
    pub(crate) crate_id: CrateId,
}

#[derive(Debug)]
pub struct CrateDependency {
    pub krate: Crate,
    pub name: Name,
}

impl Crate {
    pub fn dependencies(&self, db: &impl HirDatabase) -> Vec<CrateDependency> {
        self.dependencies_impl(db)
    }
    pub fn root_module(&self, db: &impl HirDatabase) -> Cancelable<Option<Module>> {
        self.root_module_impl(db)
    }
}
