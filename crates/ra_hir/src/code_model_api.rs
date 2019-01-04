use ra_db::{CrateId, Cancelable};

use crate::{Name, db::HirDatabase, DefId};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Module {
    pub(crate) def_id: DefId,
}

impl Module {
    /// Returns the crate this module is part of.
    pub fn krate(&self, db: &impl HirDatabase) -> Cancelable<Option<Crate>> {
        self.krate_impl(db)
    }

    pub fn crate_root(&self, db: &impl HirDatabase) -> Cancelable<Module> {
        self.crate_root_impl(db)
    }

    /// Finds a child module with the specified name.
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
    }
}
