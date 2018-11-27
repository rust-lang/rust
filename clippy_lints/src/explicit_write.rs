// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::utils::opt_def_id;
use crate::utils::{is_expn_of, match_def_path, resolve_node, span_lint};
use if_chain::if_chain;

/// **What it does:** Checks for usage of `write!()` / `writeln()!` which can be
/// replaced with `(e)print!()` / `(e)println!()`
///
/// **Why is this bad?** Using `(e)println! is clearer and more concise
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// // this would be clearer as `eprintln!("foo: {:?}", bar);`
/// writeln!(&mut io::stderr(), "foo: {:?}", bar).unwrap();
/// ```
declare_clippy_lint! {
pub EXPLICIT_WRITE,
complexity,
"using the `write!()` family of functions instead of the `print!()` family \
 of functions, when using the latter would work"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(EXPLICIT_WRITE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            // match call to unwrap
            if let ExprKind::MethodCall(ref unwrap_fun, _, ref unwrap_args) = expr.node;
            if unwrap_fun.ident.name == "unwrap";
            // match call to write_fmt
            if unwrap_args.len() > 0;
            if let ExprKind::MethodCall(ref write_fun, _, ref write_args) =
                unwrap_args[0].node;
            if write_fun.ident.name == "write_fmt";
            // match calls to std::io::stdout() / std::io::stderr ()
            if write_args.len() > 0;
            if let ExprKind::Call(ref dest_fun, _) = write_args[0].node;
            if let ExprKind::Path(ref qpath) = dest_fun.node;
            if let Some(dest_fun_id) =
                opt_def_id(resolve_node(cx, qpath, dest_fun.hir_id));
            if let Some(dest_name) = if match_def_path(cx.tcx, dest_fun_id, &["std", "io", "stdio", "stdout"]) {
                Some("stdout")
            } else if match_def_path(cx.tcx, dest_fun_id, &["std", "io", "stdio", "stderr"]) {
                Some("stderr")
            } else {
                None
            };
            then {
                let write_span = unwrap_args[0].span;
                let calling_macro =
                    // ordering is important here, since `writeln!` uses `write!` internally
                    if is_expn_of(write_span, "writeln").is_some() {
                        Some("writeln")
                    } else if is_expn_of(write_span, "write").is_some() {
                        Some("write")
                    } else {
                        None
                    };
                let prefix = if dest_name == "stderr" {
                    "e"
                } else {
                    ""
                };
                if let Some(macro_name) = calling_macro {
                    span_lint(
                        cx,
                        EXPLICIT_WRITE,
                        expr.span,
                        &format!(
                            "use of `{}!({}(), ...).unwrap()`. Consider using `{}{}!` instead",
                            macro_name,
                            dest_name,
                            prefix,
                            macro_name.replace("write", "print")
                        )
                    );
                } else {
                    span_lint(
                        cx,
                        EXPLICIT_WRITE,
                        expr.span,
                        &format!(
                            "use of `{}().write_fmt(...).unwrap()`. Consider using `{}print!` instead",
                            dest_name,
                            prefix,
                        )
                    );
                }
            }
        }
    }
}
