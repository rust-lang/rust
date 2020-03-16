# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [v0.7.1] - 2020-03-16

### Added
- Added `HashMap::get_key_value_mut`. (#145)

### Changed
- Optimized `Clone` implementation. (#146)

## [v0.7.0] - 2020-01-31

### Added
- Added a `drain_filter` function to `HashMap`. (#135)

### Changed
- Updated `ahash` dependency to 0.3. (#141)
- Optimized set union and intersection. (#130)
- `raw_entry` can now be used without requiring `S: BuildHasher`. (#123)
- `RawTable::bucket_index` can now be used under the `raw` feature. (#128)

## [v0.6.3] - 2019-10-31

### Added
- Added an `ahash-compile-time-rng` feature (enabled by default) which allows disabling the
  `compile-time-rng` feature in `ahash` to work around a Cargo bug. (#125)

## [v0.6.2] - 2019-10-23

### Added
- Added an `inline-more` feature (enabled by default) which allows choosing a tradeoff between 
  runtime performance and compilation time. (#119)

## [v0.6.1] - 2019-10-04

### Added
- Added `Entry::insert` and `RawEntryMut::insert`. (#118)

### Changed
- `Group::static_empty` was changed from a `const` to a `static` (#116).

## [v0.6.0] - 2019-08-13

### Fixed
- Fixed AHash accidentally depending on `std`. (#110)

### Changed
- The minimum Rust version has been bumped to 1.32 (due to `rand` dependency).

## ~~[v0.5.1] - 2019-08-04~~

This release was _yanked_ due to a breaking change for users of `no-default-features`.

### Added
- The experimental and unsafe `RawTable` API is available under the "raw" feature. (#108)
- Added entry-like methods for `HashSet`. (#98)

### Changed
- Changed the default hasher from FxHash to AHash. (#97)
- `hashbrown` is now fully `no_std` on recent Rust versions (1.36+). (#96)

### Fixed
- We now avoid growing the table during insertions when it wasn't necessary. (#106)
- `RawOccupiedEntryMut` now properly implements `Send` and `Sync`. (#100)
- Relaxed `lazy_static` version. (#92)

## [v0.5.0] - 2019-06-12

### Fixed
- Resize with a more conservative amount of space after deletions. (#86)

### Changed
- Exposed the Layout of the failed allocation in CollectionAllocErr::AllocErr. (#89)

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

[Unreleased]: https://github.com/rust-lang/hashbrown/compare/v0.7.1...HEAD
[v0.7.1]: https://github.com/rust-lang/hashbrown/compare/v0.7.0...v0.7.1
[v0.7.0]: https://github.com/rust-lang/hashbrown/compare/v0.6.3...v0.7.0
[v0.6.3]: https://github.com/rust-lang/hashbrown/compare/v0.6.2...v0.6.3
[v0.6.2]: https://github.com/rust-lang/hashbrown/compare/v0.6.1...v0.6.2
[v0.6.1]: https://github.com/rust-lang/hashbrown/compare/v0.6.0...v0.6.1
[v0.6.0]: https://github.com/rust-lang/hashbrown/compare/v0.5.1...v0.6.0
[v0.5.1]: https://github.com/rust-lang/hashbrown/compare/v0.5.0...v0.5.1
[v0.5.0]: https://github.com/rust-lang/hashbrown/compare/v0.4.0...v0.5.0
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
