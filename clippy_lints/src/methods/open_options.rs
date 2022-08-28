use clippy_utils::diagnostics::span_lint;
use clippy_utils::paths;
use clippy_utils::ty::match_type;
use rustc_ast::ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::{Span, Spanned};

use super::NONSENSICAL_OPEN_OPTIONS;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_method(method_id)
        && match_type(cx, cx.tcx.type_of(impl_id), &paths::OPEN_OPTIONS)
    {
        let mut options = Vec::new();
        get_open_options(cx, recv, &mut options);
        check_open_options(cx, &options, e.span);
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Argument {
    True,
    False,
    Unknown,
}

#[derive(Debug)]
enum OpenOption {
    Write,
    Read,
    Truncate,
    Create,
    Append,
}

fn get_open_options(cx: &LateContext<'_>, argument: &Expr<'_>, options: &mut Vec<(OpenOption, Argument)>) {
    if let ExprKind::MethodCall(path, arguments, _) = argument.kind {
        let obj_ty = cx.typeck_results().expr_ty(&arguments[0]).peel_refs();

        // Only proceed if this is a call on some object of type std::fs::OpenOptions
        if match_type(cx, obj_ty, &paths::OPEN_OPTIONS) && arguments.len() >= 2 {
            let argument_option = match arguments[1].kind {
                ExprKind::Lit(ref span) => {
                    if let Spanned {
                        node: LitKind::Bool(lit),
                        ..
                    } = *span
                    {
                        if lit { Argument::True } else { Argument::False }
                    } else {
                        // The function is called with a literal which is not a boolean literal.
                        // This is theoretically possible, but not very likely.
                        return;
                    }
                },
                _ => Argument::Unknown,
            };

            match path.ident.as_str() {
                "create" => {
                    options.push((OpenOption::Create, argument_option));
                },
                "append" => {
                    options.push((OpenOption::Append, argument_option));
                },
                "truncate" => {
                    options.push((OpenOption::Truncate, argument_option));
                },
                "read" => {
                    options.push((OpenOption::Read, argument_option));
                },
                "write" => {
                    options.push((OpenOption::Write, argument_option));
                },
                _ => (),
            }

            get_open_options(cx, &arguments[0], options);
        }
    }
}

fn check_open_options(cx: &LateContext<'_>, options: &[(OpenOption, Argument)], span: Span) {
    let (mut create, mut append, mut truncate, mut read, mut write) = (false, false, false, false, false);
    let (mut create_arg, mut append_arg, mut truncate_arg, mut read_arg, mut write_arg) =
        (false, false, false, false, false);
    // This code is almost duplicated (oh, the irony), but I haven't found a way to
    // unify it.

    for option in options {
        match *option {
            (OpenOption::Create, arg) => {
                if create {
                    span_lint(
                        cx,
                        NONSENSICAL_OPEN_OPTIONS,
                        span,
                        "the method `create` is called more than once",
                    );
                } else {
                    create = true;
                }
                create_arg = create_arg || (arg == Argument::True);
            },
            (OpenOption::Append, arg) => {
                if append {
                    span_lint(
                        cx,
                        NONSENSICAL_OPEN_OPTIONS,
                        span,
                        "the method `append` is called more than once",
                    );
                } else {
                    append = true;
                }
                append_arg = append_arg || (arg == Argument::True);
            },
            (OpenOption::Truncate, arg) => {
                if truncate {
                    span_lint(
                        cx,
                        NONSENSICAL_OPEN_OPTIONS,
                        span,
                        "the method `truncate` is called more than once",
                    );
                } else {
                    truncate = true;
                }
                truncate_arg = truncate_arg || (arg == Argument::True);
            },
            (OpenOption::Read, arg) => {
                if read {
                    span_lint(
                        cx,
                        NONSENSICAL_OPEN_OPTIONS,
                        span,
                        "the method `read` is called more than once",
                    );
                } else {
                    read = true;
                }
                read_arg = read_arg || (arg == Argument::True);
            },
            (OpenOption::Write, arg) => {
                if write {
                    span_lint(
                        cx,
                        NONSENSICAL_OPEN_OPTIONS,
                        span,
                        "the method `write` is called more than once",
                    );
                } else {
                    write = true;
                }
                write_arg = write_arg || (arg == Argument::True);
            },
        }
    }

    if read && truncate && read_arg && truncate_arg && !(write && write_arg) {
        span_lint(
            cx,
            NONSENSICAL_OPEN_OPTIONS,
            span,
            "file opened with `truncate` and `read`",
        );
    }
    if append && truncate && append_arg && truncate_arg {
        span_lint(
            cx,
            NONSENSICAL_OPEN_OPTIONS,
            span,
            "file opened with `append` and `truncate`",
        );
    }
}
