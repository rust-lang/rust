// compile-flags:-Z unstable-options --show-coverage
// check-pass

// The point of this test is to ensure that the number of "documented" items
// is higher than in `enum-tuple.rs`.

//! (remember the crate root is still a module)

/// so check out this enum here
pub enum ThisEnum {
    /// VarOne.
    VarOne(
        /// hello!
        String,
    ),
    /// Var Two.
    VarTwo(
        /// Hello
        String,
        /// Bis repetita.
        String,
    ),
}

/// Struct.
pub struct ThisStruct(
    /// hello
    u32,
);

/// Struct.
pub struct ThisStruct2(
    /// hello
    u32,
    /// Bis repetita.
    u8,
);
