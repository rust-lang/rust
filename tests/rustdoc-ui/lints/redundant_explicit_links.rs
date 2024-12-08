//@ run-rustfix

#![deny(rustdoc::redundant_explicit_links)]

pub fn dummy_target() {}

/// [dummy_target](dummy_target)
//~^ ERROR redundant explicit link target
/// [`dummy_target`](dummy_target)
//~^ ERROR redundant explicit link target
///
/// [Vec](Vec)
//~^ ERROR redundant explicit link target
/// [`Vec`](Vec)
//~^ ERROR redundant explicit link target
/// [Vec](std::vec::Vec)
//~^ ERROR redundant explicit link target
/// [`Vec`](std::vec::Vec)
//~^ ERROR redundant explicit link target
/// [std::vec::Vec](Vec)
//~^ ERROR redundant explicit link target
/// [`std::vec::Vec`](Vec)
//~^ ERROR redundant explicit link target
/// [std::vec::Vec](std::vec::Vec)
//~^ ERROR redundant explicit link target
/// [`std::vec::Vec`](std::vec::Vec)
//~^ ERROR redundant explicit link target
///
/// [usize](usize)
//~^ ERROR redundant explicit link target
/// [`usize`](usize)
//~^ ERROR redundant explicit link target
/// [usize](std::primitive::usize)
//~^ ERROR redundant explicit link target
/// [`usize`](std::primitive::usize)
//~^ ERROR redundant explicit link target
/// [std::primitive::usize](usize)
//~^ ERROR redundant explicit link target
/// [`std::primitive::usize`](usize)
//~^ ERROR redundant explicit link target
/// [std::primitive::usize](std::primitive::usize)
//~^ ERROR redundant explicit link target
/// [`std::primitive::usize`](std::primitive::usize)
//~^ ERROR redundant explicit link target
///
/// [dummy_target](dummy_target) TEXT
//~^ ERROR redundant explicit link target
/// [`dummy_target`](dummy_target) TEXT
//~^ ERROR redundant explicit link target
pub fn should_warn_inline() {}

/// [`Vec<T>`](Vec)
/// [`Vec<T>`](std::vec::Vec)
pub fn should_not_warn_inline() {}

/// [dummy_target][dummy_target]
//~^ ERROR redundant explicit link target
/// [`dummy_target`][dummy_target]
//~^ ERROR redundant explicit link target
///
/// [Vec][Vec]
//~^ ERROR redundant explicit link target
/// [`Vec`][Vec]
//~^ ERROR redundant explicit link target
/// [Vec][std::vec::Vec]
//~^ ERROR redundant explicit link target
/// [`Vec`][std::vec::Vec]
//~^ ERROR redundant explicit link target
/// [std::vec::Vec][Vec]
//~^ ERROR redundant explicit link target
/// [`std::vec::Vec`][Vec]
//~^ ERROR redundant explicit link target
/// [std::vec::Vec][std::vec::Vec]
//~^ ERROR redundant explicit link target
/// [`std::vec::Vec`][std::vec::Vec]
//~^ ERROR redundant explicit link target
///
/// [usize][usize]
//~^ ERROR redundant explicit link target
/// [`usize`][usize]
//~^ ERROR redundant explicit link target
/// [usize][std::primitive::usize]
//~^ ERROR redundant explicit link target
/// [`usize`][std::primitive::usize]
//~^ ERROR redundant explicit link target
/// [std::primitive::usize][usize]
//~^ ERROR redundant explicit link target
/// [`std::primitive::usize`][usize]
//~^ ERROR redundant explicit link target
/// [std::primitive::usize][std::primitive::usize]
//~^ ERROR redundant explicit link target
/// [`std::primitive::usize`][std::primitive::usize]
//~^ ERROR redundant explicit link target
///
/// [dummy_target][dummy_target] TEXT
//~^ ERROR redundant explicit link target
/// [`dummy_target`][dummy_target] TEXT
//~^ ERROR redundant explicit link target
pub fn should_warn_reference_unknown() {}

/// [`Vec<T>`][Vec]
/// [`Vec<T>`][std::vec::Vec]
pub fn should_not_warn_reference_unknown() {}

/// [dummy_target][dummy_target]
//~^ ERROR redundant explicit link target
/// [`dummy_target`][dummy_target]
//~^ ERROR redundant explicit link target
///
/// [Vec][Vec]
//~^ ERROR redundant explicit link target
/// [`Vec`][Vec]
//~^ ERROR redundant explicit link target
/// [Vec][std::vec::Vec]
//~^ ERROR redundant explicit link target
/// [`Vec`][std::vec::Vec]
//~^ ERROR redundant explicit link target
/// [std::vec::Vec][Vec]
//~^ ERROR redundant explicit link target
/// [`std::vec::Vec`][Vec]
//~^ ERROR redundant explicit link target
/// [std::vec::Vec][std::vec::Vec]
//~^ ERROR redundant explicit link target
/// [`std::vec::Vec`][std::vec::Vec]
//~^ ERROR redundant explicit link target
///
/// [usize][usize]
//~^ ERROR redundant explicit link target
/// [`usize`][usize]
//~^ ERROR redundant explicit link target
/// [usize][std::primitive::usize]
//~^ ERROR redundant explicit link target
/// [`usize`][std::primitive::usize]
//~^ ERROR redundant explicit link target
/// [std::primitive::usize][usize]
//~^ ERROR redundant explicit link target
/// [`std::primitive::usize`][usize]
//~^ ERROR redundant explicit link target
/// [std::primitive::usize][std::primitive::usize]
//~^ ERROR redundant explicit link target
/// [`std::primitive::usize`][std::primitive::usize]
//~^ ERROR redundant explicit link target
///
/// [dummy_target][dummy_target] TEXT
//~^ ERROR redundant explicit link target
/// [`dummy_target`][dummy_target] TEXT
//~^ ERROR redundant explicit link target
///
/// [dummy_target]: dummy_target
/// [Vec]: Vec
/// [std::vec::Vec]: Vec
/// [usize]: usize
/// [std::primitive::usize]: usize
pub fn should_warn_reference() {}

/// [`Vec<T>`]: Vec
/// [`Vec<T>`]: std::vec::Vec
pub fn should_not_warn_reference() {}
