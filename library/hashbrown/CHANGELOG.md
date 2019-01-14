# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/Amanieu/hashbrown/compare/v0.1.8...HEAD
[v0.1.8]: https://github.com/Amanieu/hashbrown/compare/v0.1.7...v0.1.8
[v0.1.7]: https://github.com/Amanieu/hashbrown/compare/v0.1.6...v0.1.7
[v0.1.6]: https://github.com/Amanieu/hashbrown/compare/v0.1.5...v0.1.6
[v0.1.5]: https://github.com/Amanieu/hashbrown/compare/v0.1.4...v0.1.5
[v0.1.4]: https://github.com/Amanieu/hashbrown/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/Amanieu/hashbrown/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/Amanieu/hashbrown/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/Amanieu/hashbrown/compare/v0.1.0...v0.1.1
