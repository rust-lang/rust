# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/Amanieu/hashbrown/compare/v0.1.5...HEAD
[v0.1.5]: https://github.com/Amanieu/hashbrown/compare/v0.1.4...v0.1.5
[v0.1.4]: https://github.com/Amanieu/hashbrown/compare/v0.1.3...v0.1.4
[v0.1.3]: https://github.com/Amanieu/hashbrown/compare/v0.1.2...v0.1.3
[v0.1.2]: https://github.com/Amanieu/hashbrown/compare/v0.1.1...v0.1.2
[v0.1.1]: https://github.com/Amanieu/hashbrown/compare/v0.1.0...v0.1.1
