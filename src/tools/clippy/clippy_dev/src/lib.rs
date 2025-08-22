#![feature(
    rustc_private,
    exit_status_error,
    if_let_guard,
    os_str_slice,
    os_string_truncate,
    slice_split_once
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications
)]
#![allow(clippy::missing_panics_doc)]

#[expect(unused_extern_crates, reason = "required to link to rustc crates")]
extern crate rustc_driver;
extern crate rustc_lexer;
extern crate rustc_literal_escaper;

pub mod deprecate_lint;
pub mod dogfood;
pub mod fmt;
pub mod lint;
pub mod new_lint;
pub mod release;
pub mod rename_lint;
pub mod serve;
pub mod setup;
pub mod sync;
pub mod update_lints;

mod utils;
pub use utils::{ClippyInfo, UpdateMode};
