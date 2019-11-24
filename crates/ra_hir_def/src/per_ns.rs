//! In rust, it is possible to have a value, a type and a macro with the same
//! name without conflicts.
//!
//! `PerNs` (per namespace) captures this.

use hir_expand::MacroDefId;

use crate::ModuleDefId;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PerNs {
    pub types: Option<ModuleDefId>,
    pub values: Option<ModuleDefId>,
    /// Since macros has different type, many methods simply ignore it.
    /// We can only use special method like `get_macros` to access it.
    pub macros: Option<MacroDefId>,
}

impl Default for PerNs {
    fn default() -> Self {
        PerNs { types: None, values: None, macros: None }
    }
}

impl PerNs {
    pub fn none() -> PerNs {
        PerNs { types: None, values: None, macros: None }
    }

    pub fn values(t: ModuleDefId) -> PerNs {
        PerNs { types: None, values: Some(t), macros: None }
    }

    pub fn types(t: ModuleDefId) -> PerNs {
        PerNs { types: Some(t), values: None, macros: None }
    }

    pub fn both(types: ModuleDefId, values: ModuleDefId) -> PerNs {
        PerNs { types: Some(types), values: Some(values), macros: None }
    }

    pub fn macros(macro_: MacroDefId) -> PerNs {
        PerNs { types: None, values: None, macros: Some(macro_) }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none() && self.macros.is_none()
    }

    pub fn take_types(self) -> Option<ModuleDefId> {
        self.types
    }

    pub fn take_values(self) -> Option<ModuleDefId> {
        self.values
    }

    pub fn take_macros(self) -> Option<MacroDefId> {
        self.macros
    }

    pub fn or(self, other: PerNs) -> PerNs {
        PerNs {
            types: self.types.or(other.types),
            values: self.values.or(other.values),
            macros: self.macros.or(other.macros),
        }
    }
}
