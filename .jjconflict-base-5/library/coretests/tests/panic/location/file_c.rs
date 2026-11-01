// Used for test super::location_{ord, eq}. Must be in a dedicated file.

// This is used for testing column ordering of Location, hence this ugly one-liner.
// We must fmt skip the entire containing module or else tidy will still complain.
#[rustfmt::skip]
mod no_fmt {
    use core::panic::Location;
    pub const fn one() -> &'static Location<'static> { Location::caller() }    pub const fn two() -> &'static Location<'static> { Location::caller() }    pub const fn three() -> &'static Location<'static> { Location::caller() }
}

pub use no_fmt::*;
