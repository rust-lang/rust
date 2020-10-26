#![allow(missing_debug_implementations)]

use crate::fmt::Arguments;

pub use super::v1::Alignment;
pub use crate::fmt::ArgumentV1 as Argument;

/// A command to be executed by fmt::write while processing a
/// [`crate::fmt::Arguments`].
#[derive(Copy, Clone)]
pub enum Cmd<'a> {
    /// Write a string literal as is, without any formatting.
    Str(&'static str),
    /// Set flags for the next `Format` command.
    SetFlags { fill: char, flags: u32, align: Alignment },
    /// Set the width for the next `Format` command.
    SetWidth(usize),
    /// Set the precision for the next `Format` command.
    SetPrecision(usize),
    /// Format an argument with a Display/Debug/LowerHex/etc. function.
    ///
    /// This resets the formatting options afterwards.
    Format(Argument<'a>),
}

/// The format_args!() macro uses this to construct an Arguments object.
pub fn new<'a>(commands: &'a [Cmd<'a>]) -> Arguments<'a> {
    Arguments { commands }
}
