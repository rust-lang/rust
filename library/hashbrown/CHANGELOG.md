# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## v0.1.1 - 2018-10-30

### Fixed:
- `erase_no_drop` optimization not triggering in the SSE2 implementation. (#3)
- Missing `Send` and `Sync` for hash map and iterator types. (#7)
- Bug when inserting into a table smaller than the group width. (#5)

## v0.1.0 - 2018-10-29

- Initial release

[Unreleased]: https://github.com/japaric/heapless/compare/v0.1.0...HEAD
