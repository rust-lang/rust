#![deny(rustdoc::redundant_explicit_links)]

pub fn dummy_target() {}

/// [dummy_target](dummy_target)
/// [`dummy_target`](dummy_target)
/// 
/// [Vec](Vec)
/// [`Vec`](Vec)
/// [Vec](std::vec::Vec)
/// [`Vec`](std::vec::Vec)
/// [std::vec::Vec](Vec)
/// [`std::vec::Vec`](Vec)
/// [std::vec::Vec](std::vec::Vec)
/// [`std::vec::Vec`](std::vec::Vec)
/// 
/// [usize](usize)
/// [`usize`](usize)
/// [usize](std::primitive::usize)
/// [`usize`](std::primitive::usize)
/// [std::primitive::usize](usize)
/// [`std::primitive::usize`](usize)
/// [std::primitive::usize](std::primitive::usize)
/// [`std::primitive::usize`](std::primitive::usize)
pub fn should_warn() {}

/// [`Vec<T>`](Vec)
/// [`Vec<T>`](std::vec::Vec)
pub fn should_not_warn() {}
