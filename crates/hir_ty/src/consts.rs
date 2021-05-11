//! Handling of concrete const values

/// A concrete constant value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstScalar {
    // for now, we only support the trivial case of constant evaluating the length of an array
    // Note that this is u64 because the target usize may be bigger than our usize
    Usize(u64),

    /// Case of an unknown value that rustc might know but we don't
    Unknown,
}

impl std::fmt::Display for ConstScalar {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            ConstScalar::Usize(us) => write!(fmt, "{}", us),
            ConstScalar::Unknown => write!(fmt, "_"),
        }
    }
}
