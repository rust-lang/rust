use super::imp;

#[derive(Debug)]
pub struct Priority(imp::Priority);

impl From<imp::Priority> for Priority {
    fn from(priority: imp::Priority) -> Self {
        Self(priority)
    }
}

impl From<Priority> for imp::Priority {
    fn from(priority: Priority) -> Self {
        priority.0
    }
}

#[derive(Debug)]
pub struct Affinity(pub(crate) imp::Affinity);

impl From<imp::Affinity> for Affinity {
    fn from(affinity: imp::Affinity) -> Self {
        Self(affinity)
    }
}

impl From<Affinity> for imp::Affinity {
    fn from(affinity: Affinity) -> Self {
        affinity.0
    }
}
