#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(core_intrinsics)]
#![feature(crate_visibility_modifier)]
#![feature(drain_filter)]
#![feature(in_band_lifetimes)]
#![feature(libc)]
#![feature(nll)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_quote)]
#![feature(rustc_private)]
#![feature(slice_patterns)]
#![feature(specialization)]
#![feature(stmt_expr_attributes)]

#![recursion_limit="256"]

extern crate libc;
extern crate proc_macro;

#[macro_use]
extern crate rustc;
#[macro_use]
extern crate rustc_data_structures;

pub use rmeta::{provide, provide_extern};

mod dependency_format;
mod foreign_modules;
mod link_args;
mod native_libs;
mod rmeta;

pub mod creader;
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
