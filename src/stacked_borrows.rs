use super::RangeMap;

pub type Timestamp = u64;

/// Information about a potentially mutable borrow
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Mut {
  /// A unique, mutable reference
  Uniq(Timestamp),
  /// Any raw pointer, or a shared borrow with interior mutability
  Raw,
}

/// Information about any kind of borrow
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Borrow {
  /// A mutable borrow, a raw pointer, or a shared borrow with interior mutability
  Mut(Mut),
  /// A shared borrow without interior mutability
  Frz(Timestamp)
}

/// An item in the borrow stack
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum BorStackItem {
  /// Defines which references are permitted to mutate *if* the location is not frozen
  Mut(Mut),
  /// A barrier, tracking the function it belongs to by its index on the call stack
  FnBarrier(usize)
}

impl Default for Borrow {
    fn default() -> Self {
        Borrow::Mut(Mut::Raw)
    }
}
