//! Check that we don't try to suggest reordering incompatible keywords `safe` and `unsafe` when
//! parsing things that looks like fn frontmatter/extern blocks.
//!
//! # Context
//!
//! Previously, there was some recovery logic related to misplaced keywords (e.g. `safe` and
//! `unsafe`) when we tried to parse fn frontmatter (this is what happens when trying to parse
//! something malformed like `unsafe safe extern {}` or `safe unsafe extern {}`). Unfortunately, the
//! recovery logic only really handled duplicate keywords or misplaced keywords. This meant that
//! incompatible keywords like {`unsafe`, `safe`} when used together produces some funny suggestion
//! e.g.
//!
//! ```text
//! help: `unsafe` must come before `safe`: `unsafe safe`
//! ```
//!
//! and then if you applied that suggestion, another suggestion in the recovery logic will tell you
//! to flip it back, ad infinitum.
//!
//! # References
//!
//! See <https://github.com/rust-lang/rust/issues/133586>.
//!
//! See `incompatible-safe-unsafe-keywords-extern-block-2.rs` for the `safe unsafe extern {}`
//! version.
#![crate_type = "lib"]

safe unsafe extern {}
//~^ ERROR expected one of `extern` or `fn`, found keyword `unsafe`
//~| NOTE expected one of `extern` or `fn`
//~| HELP `safe` and `unsafe` are incompatible, use only one of the keywords
