#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(libc)]
#![feature(nll)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_sort_by_cached_key)]
#![feature(crate_visibility_modifier)]
#![feature(specialization)]
#![feature(rustc_private)]

#![recursion_limit="256"]

extern crate libc;
#[macro_use]
extern crate log;
extern crate memmap;
extern crate stable_deref_trait;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate flate2;
extern crate serialize as rustc_serialize; // used by deriving
extern crate rustc_errors as errors;
extern crate syntax_ext;
extern crate proc_macro;

#[macro_use]
extern crate rustc;
extern crate rustc_target;
#[macro_use]
extern crate rustc_data_structures;

mod diagnostics;

mod index_builder;
mod index;
mod encoder;
mod decoder;
mod cstore_impl;
mod isolated_encoder;
mod schema;
mod native_libs;
mod link_args;
mod foreign_modules;

pub mod creader;
pub mod cstore;
pub mod dynamic_lib;
pub mod locator;

pub fn validate_crate_name(
    sess: Option<&rustc::session::Session>,
    s: &str,
    sp: Option<syntax_pos::Span>
) {
    let mut err_count = 0;
    {
        let mut say = |s: &str| {
            match (sp, sess) {
                (_, None) => bug!("{}", s),
                (Some(sp), Some(sess)) => sess.span_err(sp, s),
                (None, Some(sess)) => sess.err(s),
            }
            err_count += 1;
        };
        if s.is_empty() {
            say("crate name must not be empty");
        }
        for c in s.chars() {
            if c.is_alphanumeric() { continue }
            if c == '_'  { continue }
            say(&format!("invalid character `{}` in crate name: `{}`", c, s));
        }
    }

    if err_count > 0 {
        sess.unwrap().abort_if_errors();
    }
}

__build_diagnostic_array! { librustc_metadata, DIAGNOSTICS }
