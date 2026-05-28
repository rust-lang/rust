use core::panic::Location;

// Used for test super::location_{ord, eq}. Must be in a dedicated file.

pub const fn one() -> &'static Location<'static> {
    Location::caller()
}

pub const fn two() -> &'static Location<'static> {
    Location::caller()
}

pub const fn three() -> &'static Location<'static> {
    Location::caller()
}
