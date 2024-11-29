//! Check that we don't try to suggest reordering incompatible keywords `safe` and `unsafe` when
//! parsing things that looks like fn frontmatter/extern blocks.
//!
//! # References
//!
//! See <https://github.com/rust-lang/rust/issues/133586>.
//!
//! See `incompatible-safe-unsafe-keywords-extern-block-1.rs` for the `unsafe safqe extern {}`
//! version.
#![crate_type = "lib"]

unsafe safe extern {}
//~^ ERROR expected one of `extern` or `fn`, found `safe`
//~| NOTE expected one of `extern` or `fn`
//~| HELP `unsafe` and `safe` are incompatible, use only one of the keywords
