# Changelog

## [0.2.5] 2017-08-31

### Added

- Format and preserve attributes on statements (#1933).

### Fixed

- Use getters to access `Span` fields (#1899).

## [0.2.4] 2017-08-30

### Added

- Add support for `Yield` (#1928).

## [0.2.3] 2017-08-30

### Added

- `multiline_closure_forces_block` configuration option (#1898).
- `multiline_match_arm_forces_block` configuration option (#1898).
- `merge_derives` configuration option (#1910).
- `struct_remove_empty_braces` configuration option (#1930).
- Various refactorings.

### Changed

- Put single-lined block comments on the same line with list-like structure's item (#1923).
- Preserve blank line between doc comment and attribute (#1925).
- Put the opening and the closing braces of enum and struct on the same line, even when `item_brace_style = "AlwaysNextLine"` (#1930).

### Fixed

- Format attributes on `ast::ForeignItem` and take max width into account (#1916).
- Ignore empty lines when calculating the shortest indent width inside macro with braces (#1918).
- Handle tabs properly inside macro with braces (#1918).
- Fix a typo in `compute_budgets_for_args()` (#1924).
- Recover comment between keyword (`impl` and `trait`) and `{` which used to get removed (#1925).
