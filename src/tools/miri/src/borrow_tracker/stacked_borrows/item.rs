use std::fmt;

use crate::borrow_tracker::BorTag;

/// An item in the per-location borrow stack.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Item(u64);

// An Item contains 3 bitfields:
// * Bits 0-61 store a BorTag.
// * Bits 61-63 store a Permission.
// * Bit 64 stores a flag which indicates if we might have a protector.
//   This is purely an optimization: if the bit is set, the tag *might* be
//   in `protected_tags`, but if the bit is not set then the tag is definitely
//   not in `protected_tags`.
const TAG_MASK: u64 = u64::MAX >> 3;
const PERM_MASK: u64 = 0x3 << 61;
const PROTECTED_MASK: u64 = 0x1 << 63;

const PERM_SHIFT: u64 = 61;
const PROTECTED_SHIFT: u64 = 63;

impl Item {
    pub fn new(tag: BorTag, perm: Permission, protected: bool) -> Self {
        assert!(tag.get() <= TAG_MASK);
        let packed_tag = tag.get();
        let packed_perm = perm.to_bits() << PERM_SHIFT;
        let packed_protected = u64::from(protected) << PROTECTED_SHIFT;

        let new = Self(packed_tag | packed_perm | packed_protected);

        debug_assert!(new.tag() == tag);
        debug_assert!(new.perm() == perm);
        debug_assert!(new.protected() == protected);

        new
    }

    /// The pointers the permission is granted to.
    pub fn tag(self) -> BorTag {
        BorTag::new(self.0 & TAG_MASK).unwrap()
    }

    /// The permission this item grants.
    pub fn perm(self) -> Permission {
        Permission::from_bits((self.0 & PERM_MASK) >> PERM_SHIFT)
    }

    /// Whether or not there is a protector for this tag
    pub fn protected(self) -> bool {
        self.0 & PROTECTED_MASK > 0
    }

    /// Set the Permission stored in this Item
    pub fn set_permission(&mut self, perm: Permission) {
        // Clear the current set permission
        self.0 &= !PERM_MASK;
        // Write Permission::Disabled to the Permission bits
        self.0 |= perm.to_bits() << PERM_SHIFT;
    }
}

impl fmt::Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?} for {:?}]", self.perm(), self.tag())
    }
}

/// Indicates which permission is granted (by this item to some pointers)
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Permission {
    /// Grants unique mutable access.
    Unique,
    /// Grants shared mutable access.
    SharedReadWrite,
    /// Grants shared read-only access.
    SharedReadOnly,
    /// Grants no access, but separates two groups of SharedReadWrite so they are not
    /// all considered mutually compatible.
    Disabled,
}

impl Permission {
    const UNIQUE: u64 = 0;
    const SHARED_READ_WRITE: u64 = 1;
    const SHARED_READ_ONLY: u64 = 2;
    const DISABLED: u64 = 3;

    fn to_bits(self) -> u64 {
        match self {
            Permission::Unique => Self::UNIQUE,
            Permission::SharedReadWrite => Self::SHARED_READ_WRITE,
            Permission::SharedReadOnly => Self::SHARED_READ_ONLY,
            Permission::Disabled => Self::DISABLED,
        }
    }

    fn from_bits(perm: u64) -> Self {
        match perm {
            Self::UNIQUE => Permission::Unique,
            Self::SHARED_READ_WRITE => Permission::SharedReadWrite,
            Self::SHARED_READ_ONLY => Permission::SharedReadOnly,
            Self::DISABLED => Permission::Disabled,
            _ => unreachable!(),
        }
    }
}
