//! Class of an ASN.1 tag.

use super::{TagNumber, CONSTRUCTED_FLAG};
use core::fmt;

/// Class of an ASN.1 tag.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Class {
    /// `UNIVERSAL`: built-in types whose meaning is the same in all
    /// applications.
    Universal = 0b00000000,

    /// `APPLICATION`: types whose meaning is specific to an application,
    ///
    /// Types in two different applications may have the same
    /// application-specific tag and different meanings.
    Application = 0b01000000,

    /// `CONTEXT-SPECIFIC`: types whose meaning is specific to a given
    /// structured type.
    ///
    /// Context-specific tags are used to distinguish between component types
    /// with the same underlying tag within the context of a given structured
    /// type, and component types in two different structured types may have
    /// the same tag and different meanings.
    ContextSpecific = 0b10000000,

    /// `PRIVATE`: types whose meaning is specific to a given enterprise.
    Private = 0b11000000,
}

impl Class {
    /// Compute the identifier octet for a tag number of this class.
    #[allow(clippy::integer_arithmetic)]
    pub(super) fn octet(self, constructed: bool, number: TagNumber) -> u8 {
        self as u8 | number.value() | (u8::from(constructed) * CONSTRUCTED_FLAG)
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Class::Universal => "UNIVERSAL",
            Class::Application => "APPLICATION",
            Class::ContextSpecific => "CONTEXT-SPECIFIC",
            Class::Private => "PRIVATE",
        })
    }
}
