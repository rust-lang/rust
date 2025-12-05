#![feature(
    exit_status_error,
    if_let_guard,
    new_range,
    new_range_api,
    os_str_slice,
    os_string_truncate,
    pattern,
    rustc_private,
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

extern crate rustc_arena;
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

mod parse;
mod utils;

pub use self::parse::{ParseCx, new_parse_cx};
pub use self::utils::{ClippyInfo, UpdateMode};
