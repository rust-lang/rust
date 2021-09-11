// compile-flags:-Z unstable-options --show-coverage
// check-pass

//! (remember the crate root is still a module)

/// so check out this enum here
pub enum ThisEnum {
    /// No need to document the field if there is only one in a tuple variant!
    VarOne(String),
    /// But if there is more than one... still fine!
    VarTwo(String, String),
}

/// Struct.
pub struct ThisStruct(u32);

/// Struct.
pub struct ThisStruct2(u32, u8);
