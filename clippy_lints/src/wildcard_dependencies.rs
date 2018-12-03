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
use crate::syntax::{ast::*, source_map::DUMMY_SP};
use crate::utils::span_lint;

use if_chain::if_chain;
use cargo_metadata;
use semver;

/// **What it does:** Checks for wildcard dependencies in the `Cargo.toml`.
///
/// **Why is this bad?** [As the edition guide says](https://rust-lang-nursery.github.io/edition-guide/rust-2018/cargo-and-crates-io/crates-io-disallows-wildcard-dependencies.html),
/// it is highly unlikely that you work with any possible version of your dependency,
/// and wildcard dependencies would cause unnecessary breakage in the ecosystem.
///
/// **Known problems:** None.
///
/// **Example:**
///
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
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &Crate) {
        let metadata = if let Ok(metadata) = cargo_metadata::metadata(None) {
            metadata
        } else {
            span_lint(cx, WILDCARD_DEPENDENCIES, DUMMY_SP, "could not read cargo metadata");
            return;
        };

        for dep in &metadata.packages[0].dependencies {
            // VersionReq::any() does not work
            if_chain! {
                if let Ok(wildcard_ver) = semver::VersionReq::parse("*");
                if let Some(ref source) = dep.source;
                if !source.starts_with("git");
                if dep.req == wildcard_ver;
                then {
                    span_lint(
                        cx,
                        WILDCARD_DEPENDENCIES,
                        DUMMY_SP,
                        &format!("wildcard dependency for `{}`", dep.name),
                    );
                }
            }
        }
    }
}
