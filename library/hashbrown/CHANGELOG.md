# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [v0.4.0] - 2019-05-30

### Fixed
- Fixed `Send` trait bounds on `IterMut` not matching the libstd one. (#82)

## [v0.3.1] - 2019-05-30

### Fixed
- Fixed incorrect use of slice in unsafe code. (#80)

## [v0.3.0] - 2019-04-23

### Changed
- Changed shrink_to to not panic if min_capacity < capacity. (#67)

### Fixed
- Worked around emscripten bug emscripten-core/emscripten-fastcomp#258. (#66)

## [v0.2.2] - 2019-04-16

### Fixed
- Inlined non-nightly lowest_set_bit_nonzero. (#64)
- Fixed build on latest nightly. (#65)

## [v0.2.1] - 2019-04-14

### Changed
- Use for_each in map Extend and FromIterator. (#58)
- Improved worst-case performance of HashSet.is_subset. (#61)

### Fixed
- Removed incorrect debug_assert. (#60)

## [v0.2.0] - 2019-03-31

### Changed
- The code has been updated to Rust 2018 edition. This means that the minimum
  Rust version has been bumped to 1.31 (2018 edition).

### Added
- Added `insert_with_hasher` to the raw_entry API to allow `K: !(Hash + Eq)`. (#54)
- Added support for using hashbrown as the hash table implementation in libstd. (#46)

### Fixed
- Fixed cargo build with minimal-versions. (#45)
- Fixed `#[may_dangle]` attributes to match the libstd `HashMap`. (#46)
- ZST keys and values are now handled properly. (#46)

## [v0.1.8] - 2019-01-14

### Added
- Rayon parallel iterator support (#37)
- `raw_entry` support (#31)
- `#[may_dangle]` on nightly (#31)
- `try_reserve` support (#31)

### Fixed
- Fixed variance on `IterMut`. (#31)

## [v0.1.7] - 2018-12-05

### Fixed
- Fixed non-SSE version of convert_special_to_empty_and_full_to_deleted. (#32)
- Fixed overflow in rehash_in_place. (#33)

## [v0.1.6] - 2018-11-17

### Fixed
- Fixed compile error on nightly. (#29)

## [v0.1.5] - 2018-11-08

### Fixed
- Fixed subtraction overflow in generic::Group::match_byte. (#28)

## [v0.1.4] - 2018-11-04

### Fixed
- Fixed a bug in the `erase_no_drop` implementation. (#26)

## [v0.1.3] - 2018-11-01

### Added
- Serde support. (#14)

### Fixed
- Make the compiler inline functions more aggressively. (#20)

## [v0.1.2] - 2018-10-31

### Fixed
- `clear` segfaults when called on an empty table. (#13)

## [v0.1.1] - 2018-10-30

### Fixed
- `erase_no_drop` optimization not triggering in the SSE2 implementation. (#3)
- Missing `Send` and `Sync` for hash map and iterator types. (#7)
- Bug when inserting into a table smaller than the group width. (#5)

## v0.1.0 - 2018-10-29

- Initial release

[Unreleased]: https://github.com/rust-lang/hashbrown/compare/v0.4.0...HEAD
[v0.4.0]: https://github.com/rust-lang/hashbrown/compare/v0.3.1...v0.4.0
[v0.3.1]: https://github.com/rust-lang/hashbrown/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/rust-lang/hashbrown/compare/v0.2.2...v0.3.0
[v0.2.2]: https://github.com/rust-lang/hashbrown/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/rust-lang/hashbrown/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/rust-lang/hashbrown/compare/v0.1.8...v0.2.0
[v0.1.8]: https://github.com/rust-lang/hashbrown/compare/v0.1.7...v0.1.8
[v0.1.7]: https://github.com/rust-lang/hashbrown/compare/v0.1.6...v0.1.7
[v0.1.6]: https://github.com/rust-lang/hashbrown/compare/v0.1.5...v0.1.6
[v0.1.5]: https://github.com/rust-lang/hashbrown/compare/v0.1.4...v0.1.5
[v0.1.4]: https://github.com/rust-lang/hashbrown/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/rust-lang/hashbrown/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/rust-lang/hashbrown/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/rust-lang/hashbrown/compare/v0.1.0...v0.1.1
