# Changelog

## [Unreleased]

- Change option `format_doc_comment` to `format_code_in_doc_comment`.
- `use_small_heuristics` changed to be an enum and stabilised. Configuration
  options are now ready for 1.0.

## [0.99.2] 2018-08-07

### Changed

- Update rustc-ap-rustc_target to 218.0.0, rustc-ap-syntax to 218.0.0, and rustc-ap-syntax_pos to 218.0.0 5c9a2b6
- Combine function-like attributes #2900

### Fixed

- Explicitly handle semicolon after the item in statement position d96e3ca
- Fix parsing '#'-hiding of rustdoc 2eca09e

## [0.99.1] 2018-08-04

### Fixed

- fix use statements ordering when a number is present 1928ae7

## [0.99.0] 2018-08-03

- 1.0 RC release

### Changed

- Clarification in README.md 30fe66b

## [0.9.0] 2018-08-01

### Added

- Handle raw identifiers 3027c21
- Format async closure 60ce411
- Add max_width option for all heuristics c2ae39e
- Add config option `format_macro_matchers` to format the metavariable matching patterns in macros 79c5ee8
- Add config option `format_macro_bodies` to format the bodies of macros 79c5ee8
- Format exitential type fc307ff
- Support raw identifiers in struct expressions f121b1a
- Format Async block and async function 0b25f60

### Changed

- Update rustc-ap-rustc_target to 211.0.0, rustc-ap-syntax to 211.0.0, and rustc-ap-syntax_pos to 211.0.0
- Put each nested import on its own line while putting non-nested imports on the same line as much as possible 42ab258
- Respect `empty_item_single_line` config option when formatting empty impls. Put the `where` on its own line to improve readability #2771
- Strip leading `|` in match arm patterns 1d4b988
- Apply short function call heuristic to attributes 3abebf9
- Indent a match guard if the pattern is multiline be4d37d
- Change default newline style to `Native` 9d8f381
- Improve formatting of series of binop expressions a4cdb68
- Trigger an internal error if we skip formatting due to a lost comment b085113
- Refactor chain formatting #2838

### Fixed

- Do not insert spaces around braces with empty body or multiple lines 2f65852
- Allow using mixed layout with comments #2766
- Handle break labels #2726
- fix rewrite_string when a line feed is present 472a2ed
- Fix an anomaly with comments and array literals b28a0cd
- Check for comments after the `=>` in a match arm 6899471

## [0.8.0,0.8.1,0.8.2] 2018-05-28

### Added

- Use scoped attributes for skip attribute https://github.com/rust-lang/rustfmt/pull/2703

### Changed

- Comment options `wrap_comments` and `normalize_comments` are reverted back to unstable 416bc4c
- Stabilise `reorder_imports` and `reorder_modules` options 7b6d2b4
- Remove `spaces_within_parens_and_brackets` option d726492
- Stabilise shorthand options: `use_try_shorthand`, `use_field_init_shorthand`, and `force_explicit_abi` 8afe367
- Stabilise `remove_nested_parens` and set default to true a70f716
- Unstabilise `unstable_features` dd9c15a
- Remove `remove_blank_lines_at_start_or_end_of_block` option 2ee8b0e
- Update rustc-ap-syntax to 146.0.0 and rustc-ap-rustc_target to 146.0.0 2c275a2
- Audit the public API #2639

### Fixed

- Handle code block in doc comment without rust prefix f1974e2

## [0.7.0] 2018-05-14

### Added

- Add integration tests against crates in the rust-lang-nursery c79f39a

### Changed

- Update rustc-ap-syntax to 128.0.0 and ustc-ap-rustc_target to 128.0.0 195395f
- Put operands on its own line when each fits in a single line f8439ce
- Improve CLI options 55ac062 1869888 798bffb 4d9de48 eca7796 8396da1 5d9f5aa

### Fixed

- Use correct line width for list attribute 61a401a
- Avoid flip-flopping impl items when reordering them 37c216c
- Formatting breaks short lines when max_width is less than 100 9b36156
- Fix variant "Mixed" of imports_layout option 8c8676c
- Improve handling of long lines f885039
- Fix up lines exceeding max width 51c07f4
- Fix handling of modules in non_modrs_mods style cf573e8
- Do not duplicate attributes on use items e59ceaf
- Do not insert an extra brace in macros with native newlines 4c9ef93

## [0.6.1] 2018-05-01

### Changed

- Change the default value of imports_indent to IndentStyle::Block https://github.com/rust-lang/rustfmt/pull/2662

### Fixed

- Handle formatting of auto traits 5b5a72c
- Use consistent formatting for empty enum and struct https://github.com/rust-lang/rustfmt/pull/2656

## [0.6.0] 2018-04-20

### Changed

- Improve public API 8669004

## [0.5.0] 2018-04-20

### Added

- Add `verbose-diff` CLI option 5194984

### Changed

- Update rustc-ap-syntax to 103.0.0 dd807e2
- Refactor to make a sensible public API ca610d3

### Fixed

- Add spaces between consecutive `..` `..=` 61d29eb

## [0.4.2] 2018-04-12

### Added

- Handle binary operators and lifetimes 0fd174d
- Add reorder_impl_items config option 94f5a05
- Add `--unstable-features` CLI option to list unstable options from the `--help` output 8208f8a
- Add merge_imports config option 5dd203e

### Changed

- Format macro arguments with vertical layout ec71459
- Reorder imports by default 164cf7d
- Do not collapse block around expr with condition on match arm 5b9b7d5
- Use vertical layout for complex attributes c77708f
- Format array using heuristics for function calls 98c6f7b
- Implement stable ordering for impl items with the the following item priority: type, const, macro, then method fa80ddf
- Reorder imports by default 164cf7d
- Group `extern crate` by default 3a138a2
- Make `error_on_line_overflow` false by default f146711
- Merge imports with the same prefix into a single nested import 1954513
- Squash the various 'reorder imports' option into one 911395a

### Fixed

- Print version is missing the channel ca6fc67
- Do not add the beginning vert to the match arm 1e1d9d4
- Follow indent style config when formatting attributes efd295a
- Do not insert newline when item is empty a8022f3
- Do not indent or unindent inside string literal ec1907b

## [0.4.1] 2018-03-16

### Added

- Add `ignore` configuration option.
- Add `license_template_path` configuration option.
- Format `lazy_static!`.

### Fixed

- Fix formatting bugs.
- Fix setting `reorder_modules` removing inline modules.
- Format attributes on block expressions.
- Support `dyn trait` syntax.
- Support multiple patterns in `if let` and `while let`.
- Support a pattern with parentheses.

## [0.4.0] 2018-03-02

### Changed

- Do not print verbose outputs when formatting with stdin.
- Preserve trailing whitespaces in doc comments.
- Scale the values of width heuristics by `max_width`.

### Fixed

- Do not reorder items with `#[macro_use]`.
- Fix formatting bugs.
- Support the beginning `|` on a match arm.

## [0.3.8] 2018-02-04

### Added

- Format (or at least try to format) `macro_rules!`.

## [0.3.7] 2018-02-01

### Added

- Add `use_field_init_shorthand` config option.
- Add `reorder_modules` configuration option.

## [0.3.6] 2018-01-18

### Fixed

- Fix panicking on formatting certain macros (#2371).

## [0.3.5] 2018-01-15

### Changed

- Format code block in comments when `wrap_comments` is set to `true`.
- Remove `same_line_attributes` configuration option.
- Rename `git-fmt` to `git-rustfmt`.

### Fixed

- Rustup to `rustc 1.25.0-nightly (e6072a7b3 2018-01-13)`.
- Fix formatting bugs.

## [0.3.4] 2017-12-23

### Added

- Add `--version` flag to `cargo-fmt`, allow `cargo fmt --version`.

### Fixed

- Rustup to `rustc 1.24.0-nightly (5165ee9e2 2017-12-22)`.

## [0.3.3] 2017-12-22

### Added

- Format trait aliases.

### Changed

- `cargo fmt` will format every workspace member.

### Fixed

- Rustup to `rustc 1.24.0-nightly (250b49205 2017-12-21)`
- Fix formatting bugs.

## [0.3.2] 2017-12-15

### Changed

- Warn when unknown configuration option is used.

### Fixed

- Rustup to `rustc 1.24.0-nightly (0077d128d 2017-12-14)`.

## [0.3.1] 2017-12-11

### Added

- Add `error_on_unformatted` configuration option.
- Add `--error-on-unformatted` command line option.

### Changed

- Do not report formatting errors on comments or strings by default.
- Rename `error_on_line_overflow_comments` to `error_on_unformatted`.

### Fixed

- Fix formatting bugs.
- Fix adding a trailing whitespace inside code block when `wrap_comments = true`.

## [0.3.0] 2017-12-11

### Added

- Support nested imports.

### Changed

- Do not report errors on skipped items.
- Do not format code block inside comments when `wrap_comments = true`.
- Keep vertical spaces between items within range.
- Format `format!` and its variants using compressed style.
- Format `write!` and its variants using compressed style.
- Format **simple** array using compressed style.

### Fixed

- Fix `rustfmt --package package_name` not working properly.
- Fix formatting bugs.

## [0.2.17] 2017-12-03

### Added

- Add `blank_lines_lower_bound` and `blank_lines_upper_bound` configuration options.

### Changed

- Combine configuration options related to width heuristic into `width_heuristic`.
- If the match arm's body is `if` expression, force to use block.

### Fixed

- Fix `cargo fmt --all` being trapped in an infinite loop.
- Fix many formatting bugs.

### Removed

- Remove legacy configuration options.

## [0.2.16] 2017-11-21

### Added

- Remove empty lines at the beginning of the file.
- Soft wrapping on doc comments.

### Changed

- Break before `|` when using multiple lines for match arm patterns.
- Combine `control_style`, `where_style` and `*_indent` config options into `indent_style`.
- Combine `item_brace_style` and `fn_brace_style` config options into `brace_style`.
- Combine config options related spacing around colons into `space_before_colon` and `space_after_colon`.

### Fixed

- Fix many bugs.

## [0.2.15] 2017-11-08

### Added

- Add git-fmt tool
- `where_single_line` configuration option.

### Changed

- Rename `chain_one_line_max` to `chain_width`.
- Change the suffix of indent-related configuration options to `_indent`.

## [0.2.14] 2017-11-06

### Fixed

- Rustup to the latest nightly.

## [0.2.13] 2017-10-30

### Fixed

- Rustup to the latest nightly.

## [0.2.12] 2017-10-29

### Fixed

- Fix a bug that `cargo fmt` hangs forever.

## [0.2.11] 2017-10-29

### Fixed

- Fix a bug that `cargo fmt` crashes.

## [0.2.10] 2017-10-28

## [0.2.9] 2017-10-16

## [0.2.8] 2017-09-28

## [0.2.7] 2017-09-21

### Added

- `binop_separator` configuration option (#1964).

### Changed

- Use horizontal layout for function call with a single argument.

### Fixed

- Fix panicking when calling `cargo fmt --all` (#1963).
- Refactorings & faster rustfmt.

## [0.2.6] 2017-09-14

### Fixed

- Fix a performance issue with nested block (#1940).
- Refactorings & faster rustfmt.

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
