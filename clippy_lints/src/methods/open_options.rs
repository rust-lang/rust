use rustc_data_structures::fx::FxHashMap;

use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;
use rustc_span::{sym, Span};

use super::{NONSENSICAL_OPEN_OPTIONS, SUSPICIOUS_OPEN_OPTIONS};

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_method(method_id)
        && is_type_diagnostic_item(cx, cx.tcx.type_of(impl_id).instantiate_identity(), sym::FsOpenOptions)
    {
        let mut options = Vec::new();
        get_open_options(cx, recv, &mut options);
        check_open_options(cx, &options, e.span);
    }
}

#[derive(Eq, PartialEq, Clone)]
enum Argument {
    Set(bool),
    Unknown,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
enum OpenOption {
    Append,
    Create,
    CreateNew,
    Read,
    Truncate,
    Write,
}
impl std::fmt::Display for OpenOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenOption::Append => write!(f, "append"),
            OpenOption::Create => write!(f, "create"),
            OpenOption::CreateNew => write!(f, "create_new"),
            OpenOption::Read => write!(f, "read"),
            OpenOption::Truncate => write!(f, "truncate"),
            OpenOption::Write => write!(f, "write"),
        }
    }
}

fn get_open_options(cx: &LateContext<'_>, argument: &Expr<'_>, options: &mut Vec<(OpenOption, Argument)>) {
    if let ExprKind::MethodCall(path, receiver, arguments, _) = argument.kind {
        let obj_ty = cx.typeck_results().expr_ty(receiver).peel_refs();

        // Only proceed if this is a call on some object of type std::fs::OpenOptions
        if is_type_diagnostic_item(cx, obj_ty, sym::FsOpenOptions) && !arguments.is_empty() {
            let argument_option = match arguments[0].kind {
                ExprKind::Lit(span) => {
                    if let Spanned {
                        node: LitKind::Bool(lit),
                        ..
                    } = span
                    {
                        Argument::Set(*lit)
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
                "create_new" => {
                    options.push((OpenOption::CreateNew, argument_option));
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

            get_open_options(cx, receiver, options);
        }
    }
}

fn check_open_options(cx: &LateContext<'_>, settings: &[(OpenOption, Argument)], span: Span) {
    // The args passed to these methods, if they have been called
    let mut options = FxHashMap::default();
    for (option, arg) in settings {
        if options.insert(option.clone(), arg.clone()).is_some() {
            span_lint(
                cx,
                NONSENSICAL_OPEN_OPTIONS,
                span,
                &format!("the method `{}` is called more than once", &option),
            );
        }
    }

    if let (Some(Argument::Set(true)), Some(Argument::Set(true))) =
        (options.get(&OpenOption::Read), options.get(&OpenOption::Truncate))
    {
        if options.get(&OpenOption::Write).unwrap_or(&Argument::Set(false)) == &Argument::Set(false) {
            span_lint(
                cx,
                NONSENSICAL_OPEN_OPTIONS,
                span,
                "file opened with `truncate` and `read`",
            );
        }
    }

    if let (Some(Argument::Set(true)), Some(Argument::Set(true))) =
        (options.get(&OpenOption::Append), options.get(&OpenOption::Truncate))
    {
        if options.get(&OpenOption::Write).unwrap_or(&Argument::Set(false)) == &Argument::Set(false) {
            span_lint(
                cx,
                NONSENSICAL_OPEN_OPTIONS,
                span,
                "file opened with `append` and `truncate`",
            );
        }
    }

    if let (Some(Argument::Set(true)), None) = (options.get(&OpenOption::Create), options.get(&OpenOption::Truncate)) {
        span_lint(
            cx,
            SUSPICIOUS_OPEN_OPTIONS,
            span,
            "file opened with `create`, but `truncate` behavior not defined",
        );
    }
}
