#[macro_use]
extern crate rustc;
extern crate syntax_pos;

use rustc::session::Session;
use syntax_pos::Span;

pub fn validate_crate_name(sess: Option<&Session>, s: &str, sp: Option<Span>) {
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
