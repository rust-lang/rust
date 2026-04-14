//! Generic envelope for things in the system.

use crate::wire::ThingId;

/// A generic envelope that provides identity.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Thing<T> {
    pub id: ThingId,
    pub value: T,
}

impl<T> Thing<T> {
    /// Create a new Thing with a default (zeroed) handle.
    pub fn new(value: T) -> Self {
        Self {
            id: ThingId::default(),
            value,
        }
    }

    /// Create a new Thing with a specific ID.
    pub fn with_id(id: ThingId, value: T) -> Self {
        Self { id, value }
    }
}
