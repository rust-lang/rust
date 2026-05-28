use std::fmt;

/// Wrapper type for formatting a `T` using its `LowerHex` implementation.
#[derive(Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hex<T>(pub T);

impl<T: fmt::LowerHex> fmt::Debug for Hex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nibbles = 2 * std::mem::size_of::<T>();
        write!(f, "0x{:0width$x}", self.0, width = nibbles)
    }
}

impl<T: fmt::LowerHex> fmt::Display for Hex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Wrapper type for formatting a `char` using `escape_default`.
#[derive(Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CharEscape(pub char);

impl fmt::Debug for CharEscape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "'{}'", self.0.escape_default())
    }
}

impl fmt::Display for CharEscape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}
