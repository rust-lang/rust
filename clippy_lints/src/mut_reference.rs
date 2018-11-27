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
use crate::rustc::ty::subst::Subst;
use crate::rustc::ty::{self, Ty};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::utils::span_lint;

/// **What it does:** Detects giving a mutable reference to a function that only
/// requires an immutable reference.
///
/// **Why is this bad?** The immutable reference rules out all other references
/// to the value. Also the code misleads about the intent of the call site.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// my_vec.push(&mut value)
/// ```
declare_clippy_lint! {
pub UNNECESSARY_MUT_PASSED,
style,
"an argument passed as a mutable reference although the callee only demands an \
 immutable reference"
}

#[derive(Copy, Clone)]
pub struct UnnecessaryMutPassed;

impl LintPass for UnnecessaryMutPassed {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNECESSARY_MUT_PASSED)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnnecessaryMutPassed {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        match e.node {
            ExprKind::Call(ref fn_expr, ref arguments) => {
                if let ExprKind::Path(ref path) = fn_expr.node {
                    check_arguments(
                        cx,
                        arguments,
                        cx.tables.expr_ty(fn_expr),
                        &print::to_string(print::NO_ANN, |s| s.print_qpath(path, false)),
                    );
                }
            },
            ExprKind::MethodCall(ref path, _, ref arguments) => {
                let def_id = cx.tables.type_dependent_defs()[e.hir_id].def_id();
                let substs = cx.tables.node_substs(e.hir_id);
                let method_type = cx.tcx.type_of(def_id).subst(cx.tcx, substs);
                check_arguments(cx, arguments, method_type, &path.ident.as_str())
            },
            _ => (),
        }
    }
}

fn check_arguments<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, arguments: &[Expr], type_definition: Ty<'tcx>, name: &str) {
    match type_definition.sty {
        ty::FnDef(..) | ty::FnPtr(_) => {
            let parameters = type_definition.fn_sig(cx.tcx).skip_binder().inputs();
            for (argument, parameter) in arguments.iter().zip(parameters.iter()) {
                match parameter.sty {
                    ty::Ref(_, _, MutImmutable)
                    | ty::RawPtr(ty::TypeAndMut {
                        mutbl: MutImmutable, ..
                    }) => {
                        if let ExprKind::AddrOf(MutMutable, _) = argument.node {
                            span_lint(
                                cx,
                                UNNECESSARY_MUT_PASSED,
                                argument.span,
                                &format!("The function/method `{}` doesn't need a mutable reference", name),
                            );
                        }
                    },
                    _ => (),
                }
            }
        },
        _ => (),
    }
}
