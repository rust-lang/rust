use rustc::lint::*;
use rustc_front::hir::{Expr, ExprMethodCall, ExprLit};
use utils::{walk_ptrs_ty_depth, match_type, span_lint, OPEN_OPTIONS_PATH};
use syntax::codemap::{Span, Spanned};
use syntax::ast::Lit_::LitBool;

/// **What it does:** This lint checks for duplicate open options as well as combinations that make no sense. It is `Warn` by default.
///
/// **Why is this bad?** In the best case, the code will be harder to read than necessary. I don't know the worst case.
///
/// **Known problems:** None
///
/// **Example:** `OpenOptions::new().read(true).truncate(true)`
declare_lint! {
    pub NONSENSICAL_OPEN_OPTIONS,
    Warn,
    "nonsensical combination of options for opening a file"
}


#[derive(Copy,Clone)]
pub struct NonSensicalOpenOptions;

impl LintPass for NonSensicalOpenOptions {
    fn get_lints(&self) -> LintArray {
        lint_array!(NONSENSICAL_OPEN_OPTIONS)
    }
}

impl LateLintPass for NonSensicalOpenOptions {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprMethodCall(ref name, _, ref arguments) = e.node {
            let (obj_ty, _) = walk_ptrs_ty_depth(cx.tcx.expr_ty(&arguments[0]));
            if name.node.as_str() == "open" && match_type(cx, obj_ty, &OPEN_OPTIONS_PATH){
                let mut options = Vec::new();
                get_open_options(cx, &arguments[0], &mut options);
                check_open_options(cx, &options, e.span);
            }
        }
    }
}

#[derive(Debug)]
enum Argument {
    True,
    False,
    Unknown
}

#[derive(Debug)]
enum OpenOption {
    Write,
    Read,
    Truncate,
    Create,
    Append
}

fn get_open_options(cx: &LateContext, argument: &Expr, options: &mut Vec<(OpenOption, Argument)>) {
    if let ExprMethodCall(ref name, _, ref arguments) = argument.node {
        let (obj_ty, _) = walk_ptrs_ty_depth(cx.tcx.expr_ty(&arguments[0]));
        
        // Only proceed if this is a call on some object of type std::fs::OpenOptions
        if match_type(cx, obj_ty, &OPEN_OPTIONS_PATH) && arguments.len() >= 2 {
            
            let argument_option = match arguments[1].node {
                ExprLit(ref span) => {
                    if let Spanned {node: LitBool(lit), ..} = **span {
                        if lit {Argument::True} else {Argument::False}
                    } else {
                        return; // The function is called with a literal
                                // which is not a boolean literal. This is theoretically
                                // possible, but not very likely.
                    }
                }
                _ => {
                    Argument::Unknown
                }
            };
            
            match &*name.node.as_str() {
                "create" => {
                    options.push((OpenOption::Create, argument_option));
                }
                "append" => {
                    options.push((OpenOption::Append, argument_option));
                }
                "truncate" => {
                    options.push((OpenOption::Truncate, argument_option));
                }
                "read" => {
                    options.push((OpenOption::Read, argument_option));
                }
                "write" => {
                    options.push((OpenOption::Write, argument_option));
                }
                _ => {}
            }
            
            get_open_options(cx, &arguments[0], options);
        }
    }
}

fn check_for_duplicates(cx: &LateContext, options: &[(OpenOption, Argument)], span: Span) {
    // This code is almost duplicated (oh, the irony), but I haven't found a way to unify it.
    if options.iter().filter(|o| if let (OpenOption::Create, _) = **o {true} else {false}).count() > 1 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "The method \"create\" \
                                                       is called more than once");
    }
    if options.iter().filter(|o| if let (OpenOption::Append, _) = **o {true} else {false}).count() > 1 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "The method \"append\" \
                                                       is called more than once");
    }
    if options.iter().filter(|o| if let (OpenOption::Truncate, _) = **o {true} else {false}).count() > 1 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "The method \"truncate\" \
                                                       is called more than once");
    }
    if options.iter().filter(|o| if let (OpenOption::Read, _) = **o {true} else {false}).count() > 1 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "The method \"read\" \
                                                       is called more than once");
    }
    if options.iter().filter(|o| if let (OpenOption::Write, _) = **o {true} else {false}).count() > 1 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "The method \"write\" \
                                                       is called more than once");
    }
}

fn check_for_inconsistencies(cx: &LateContext, options: &[(OpenOption, Argument)], span: Span) {
    // Truncate + read makes no sense.
    if options.iter().filter(|o| if let (OpenOption::Read, Argument::True) = **o {true} else {false}).count() > 0 &&
       options.iter().filter(|o| if let (OpenOption::Truncate, Argument::True) = **o {true} else {false}).count() > 0 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "File opened with \"truncate\" and \"read\"");
    }
    
    // Append + truncate makes no sense.
    if options.iter().filter(|o| if let (OpenOption::Append, Argument::True) = **o {true} else {false}).count() > 0 &&
       options.iter().filter(|o| if let (OpenOption::Truncate, Argument::True) = **o {true} else {false}).count() > 0 {
        span_lint(cx, NONSENSICAL_OPEN_OPTIONS, span, "File opened with \"append\" and \"truncate\"");
    }
}

fn check_open_options(cx: &LateContext, options: &[(OpenOption, Argument)], span: Span) {
    check_for_duplicates(cx, options, span);
    check_for_inconsistencies(cx, options, span);
}
