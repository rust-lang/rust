// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::syntax::ast::*;
use crate::utils::span_lint;

use cargo_metadata;
use lazy_static::lazy_static;
use semver;

/// **What it does:** Checks to see if wildcard dependencies are being used.
///
/// **Why is this bad?** [As the edition guide sais](https://rust-lang-nursery.github.io/edition-guide/rust-2018/cargo-and-crates-io/crates-io-disallows-wildcard-dependencies.html),
/// it is highly unlikely that you work with any possible version of your dependency,
/// and wildcard dependencies would cause unnecessary breakage in the ecosystem.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```toml
/// [dependencies]
/// regex = "*"
/// ```
declare_clippy_lint! {
    pub WILDCARD_DEPENDENCIES,
    cargo,
    "wildcard dependencies being used"
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(WILDCARD_DEPENDENCIES)
    }
}

impl EarlyLintPass for Pass {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        let metadata = if let Ok(metadata) = cargo_metadata::metadata(None) {
            metadata
        } else {
            span_lint(cx, WILDCARD_DEPENDENCIES, krate.span, "could not read cargo metadata");
            return;
        };

        lazy_static! {
            static ref WILDCARD_VERSION_REQ: semver::VersionReq = semver::VersionReq::parse("*").unwrap();
        }

        for dep in &metadata.packages[0].dependencies {
            if dep.req == *WILDCARD_VERSION_REQ {
                span_lint(
                    cx,
                    WILDCARD_DEPENDENCIES,
                    krate.span,
                    &format!("wildcard dependency for `{}`", dep.name),
                );
            }
        }
    }
}
