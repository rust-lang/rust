#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(in_band_lifetimes)]
#![feature(libc)]
#![feature(nll)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(crate_visibility_modifier)]
#![feature(specialization)]
#![feature(rustc_private)]

#![recursion_limit="256"]

#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

extern crate libc;
#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize; // used by deriving
extern crate proc_macro;

#[macro_use]
extern crate rustc;
#[macro_use]
extern crate rustc_data_structures;

mod error_codes;

mod index;
mod encoder;
mod decoder;
mod cstore_impl;
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
