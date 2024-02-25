//@ compile-flags:-Z unstable-options --show-coverage
//@ check-pass

//! (remember the crate root is still a module)

/// so check out this enum here
pub enum ThisEnum {
    /// this variant has some weird stuff going on
    VarOne {
        /// like, it has some named fields inside
        field_one: usize,
        // (these show up as struct fields)
        field_two: usize,
    },
    /// here's another variant for you
    VarTwo(String),
    // but not all of them need to be documented as thoroughly
    VarThree,
}

/// uninhabited enums? sure, let's throw one of those around
pub enum OtherEnum {}
