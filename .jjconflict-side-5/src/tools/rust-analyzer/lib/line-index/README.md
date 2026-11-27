# line-index

This crate is developed as part of `rust-analyzer`.

line-index is a library to convert between text offsets and corresponding line/column coordinates.

## Installation

To add this crate to a project simply run `cargo add line-index`.

## Usage

The main structure is `LineIndex`.

It is constructed with an UTF-8 string, but also supports UTF-16 and UTF-32 offsets.

### Example

```rust
use line_index::LineIndex;

let line_index = LineIndex::new("This is a\nmulti-line\ntext.");
line_index.line_col(3.into()); // LineCol { line: 0, col: 3 }
line_index.line_col(13.into()); // LineCol { line: 1, col: 3 }
line_index.offset(LineCol { line: 2, col: 3 }); // Some (24)
```

## SemVer

This crate uses [semver](https://semver.org/) versioning.
