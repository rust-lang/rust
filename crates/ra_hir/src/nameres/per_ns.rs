use crate::MacroDef;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Namespace {
    Types,
    Values,
    // Note that only type inference uses this enum, and it doesn't care about macros.
    // Macro,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PerNs<T> {
    pub types: Option<T>,
    pub values: Option<T>,
    /// Since macros has different type, many methods simply ignore it.
    /// We can only use special method like `get_macros` to access it.
    pub macros: Option<MacroDef>,
}

impl<T> Default for PerNs<T> {
    fn default() -> Self {
        PerNs { types: None, values: None, macros: None }
    }
}

impl<T> PerNs<T> {
    pub fn none() -> PerNs<T> {
        PerNs { types: None, values: None, macros: None }
    }

    pub fn values(t: T) -> PerNs<T> {
        PerNs { types: None, values: Some(t), macros: None }
    }

    pub fn types(t: T) -> PerNs<T> {
        PerNs { types: Some(t), values: None, macros: None }
    }

    pub fn both(types: T, values: T) -> PerNs<T> {
        PerNs { types: Some(types), values: Some(values), macros: None }
    }

    pub fn macros(macro_: MacroDef) -> PerNs<T> {
        PerNs { types: None, values: None, macros: Some(macro_) }
    }

    pub fn is_none(&self) -> bool {
        self.types.is_none() && self.values.is_none() && self.macros.is_none()
    }

    pub fn is_all(&self) -> bool {
        self.types.is_some() && self.values.is_some() && self.macros.is_some()
    }

    pub fn take_types(self) -> Option<T> {
        self.types
    }

    pub fn take_values(self) -> Option<T> {
        self.values
    }

    pub fn get_macros(&self) -> Option<MacroDef> {
        self.macros
    }

    pub fn only_macros(&self) -> PerNs<T> {
        PerNs { types: None, values: None, macros: self.macros }
    }

    pub fn or(self, other: PerNs<T>) -> PerNs<T> {
        PerNs {
            types: self.types.or(other.types),
            values: self.values.or(other.values),
            macros: self.macros.or(other.macros),
        }
    }
}
