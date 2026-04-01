/// Flags to be passed to the MapMemory struct.
/// Note that it is an error to have memory be
/// writable and not readable.
#[derive(Copy, PartialEq, Eq, Clone, PartialOrd, Ord, Hash, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct MemoryFlags {
    bits: usize,
}

impl MemoryFlags {
    /// Free this memory
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const FREE: Self = Self { bits: 0b0000_0000 };

    /// Immediately allocate this memory.  Otherwise it will
    /// be demand-paged.  This is implicitly set when `phys`
    /// is not 0.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const RESERVE: Self = Self { bits: 0b0000_0001 };

    /// Allow the CPU to read from this page.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const R: Self = Self { bits: 0b0000_0010 };

    /// Allow the CPU to write to this page.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const W: Self = Self { bits: 0b0000_0100 };

    /// Allow the CPU to execute from this page.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const X: Self = Self { bits: 0b0000_1000 };

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn bits(&self) -> usize {
        self.bits
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_bits(raw: usize) -> Option<MemoryFlags> {
        if raw > 16 { None } else { Some(MemoryFlags { bits: raw }) }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn empty() -> MemoryFlags {
        MemoryFlags { bits: 0 }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn all() -> MemoryFlags {
        MemoryFlags { bits: 15 }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::fmt::Binary for MemoryFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Binary::fmt(&self.bits, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::fmt::Octal for MemoryFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Octal::fmt(&self.bits, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::fmt::LowerHex for MemoryFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::LowerHex::fmt(&self.bits, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::fmt::UpperHex for MemoryFlags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::UpperHex::fmt(&self.bits, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::BitOr for MemoryFlags {
    type Output = Self;

    /// Returns the union of the two sets of flags.
    #[inline]
    fn bitor(self, other: MemoryFlags) -> Self {
        Self { bits: self.bits | other.bits }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::BitOrAssign for MemoryFlags {
    /// Adds the set of flags.
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        self.bits |= other.bits;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::BitXor for MemoryFlags {
    type Output = Self;

    /// Returns the left flags, but with all the right flags toggled.
    #[inline]
    fn bitxor(self, other: Self) -> Self {
        Self { bits: self.bits ^ other.bits }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::BitXorAssign for MemoryFlags {
    /// Toggles the set of flags.
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        self.bits ^= other.bits;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::BitAnd for MemoryFlags {
    type Output = Self;

    /// Returns the intersection between the two sets of flags.
    #[inline]
    fn bitand(self, other: Self) -> Self {
        Self { bits: self.bits & other.bits }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::BitAndAssign for MemoryFlags {
    /// Disables all flags disabled in the set.
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        self.bits &= other.bits;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::Sub for MemoryFlags {
    type Output = Self;

    /// Returns the set difference of the two sets of flags.
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { bits: self.bits & !other.bits }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::SubAssign for MemoryFlags {
    /// Disables all flags enabled in the set.
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.bits &= !other.bits;
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl core::ops::Not for MemoryFlags {
    type Output = Self;

    /// Returns the complement of this set of flags.
    #[inline]
    fn not(self) -> Self {
        Self { bits: !self.bits } & MemoryFlags { bits: 15 }
    }
}
