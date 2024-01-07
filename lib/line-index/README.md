# line-index

This crate is developped as part of `rust-analyzer`.

line-index is a library to convert between text offset and its corresponding line/column.

## Installation

To add this crate to a project simply run `cargo add line-index`.

## Usage

The main structure is `LineIndex`. It is constructed with an utf-8 text then various utility functions can be used on it.

### Example

```rust
use line_index::LineIndex;

let line_index = LineIndex::new("This is a\nmulti-line\ntext.");
line_index.line_col(3.into()); // LineCol { line: 0, col: 3 }
line_index.line_col(13.into()); // LineCol { line: 1, col: 3 }
line_index.offset(LineCol { line: 2, col: 3 }); // Some (24)
```

## SemVer

This crate follows [semver principles]([url](https://semver.org/)https://semver.org/).
