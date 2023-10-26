use rustc_data_structures::fx::FxHashMap;

use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::paths;
use clippy_utils::ty::{is_type_diagnostic_item, match_type};
use rustc_ast::ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::source_map::Spanned;
use rustc_span::{sym, Span};

use super::{NONSENSICAL_OPEN_OPTIONS, SUSPICIOUS_OPEN_OPTIONS};

fn is_open_options(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    is_type_diagnostic_item(cx, ty, sym::FsOpenOptions) || match_type(cx, ty, &paths::TOKIO_IO_OPEN_OPTIONS)
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_method(method_id)
        && is_open_options(cx, cx.tcx.type_of(impl_id).instantiate_identity())
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

fn get_open_options(cx: &LateContext<'_>, argument: &Expr<'_>, options: &mut Vec<(OpenOption, Argument, Span)>) {
    if let ExprKind::MethodCall(path, receiver, arguments, span) = argument.kind {
        let obj_ty = cx.typeck_results().expr_ty(receiver).peel_refs();

        // Only proceed if this is a call on some object of type std::fs::OpenOptions
        if !arguments.is_empty() && is_open_options(cx, obj_ty) {
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
                    options.push((OpenOption::Create, argument_option, span));
                },
                "create_new" => {
                    options.push((OpenOption::CreateNew, argument_option, span));
                },
                "append" => {
                    options.push((OpenOption::Append, argument_option, span));
                },
                "truncate" => {
                    options.push((OpenOption::Truncate, argument_option, span));
                },
                "read" => {
                    options.push((OpenOption::Read, argument_option, span));
                },
                "write" => {
                    options.push((OpenOption::Write, argument_option, span));
                },
                _ => (),
            }

            get_open_options(cx, receiver, options);
        }
    }
}

fn check_open_options(cx: &LateContext<'_>, settings: &[(OpenOption, Argument, Span)], span: Span) {
    // The args passed to these methods, if they have been called
    let mut options = FxHashMap::default();
    for (option, arg, sp) in settings {
        if let Some((_, prev_span)) = options.insert(option.clone(), (arg.clone(), *sp)) {
            span_lint(
                cx,
                NONSENSICAL_OPEN_OPTIONS,
                prev_span,
                &format!("the method `{}` is called more than once", &option),
            );
        }
    }

    if let Some((Argument::Set(true), _)) = options.get(&OpenOption::Read)
        && let Some((Argument::Set(true), _)) = options.get(&OpenOption::Truncate)
        && let None | Some((Argument::Set(false), _)) = options.get(&OpenOption::Write)
    {
        span_lint(
            cx,
            NONSENSICAL_OPEN_OPTIONS,
            span,
            "file opened with `truncate` and `read`",
        );
    }

    if let Some((Argument::Set(true), _)) = options.get(&OpenOption::Append)
        && let Some((Argument::Set(true), _)) = options.get(&OpenOption::Truncate)
    {
        span_lint(
            cx,
            NONSENSICAL_OPEN_OPTIONS,
            span,
            "file opened with `append` and `truncate`",
        );
    }

    if let Some((Argument::Set(true), create_span)) = options.get(&OpenOption::Create)
        && let None = options.get(&OpenOption::Truncate)
    {
        span_lint_and_then(
            cx,
            SUSPICIOUS_OPEN_OPTIONS,
            *create_span,
            "file opened with `create`, but `truncate` behavior not defined",
            |diag| {
                diag
                    .span_suggestion(create_span.shrink_to_hi(), "add", ".truncate(true)".to_string(), rustc_errors::Applicability::MaybeIncorrect)
                    .help("if you intend to overwrite an existing file entirely, call `.truncate(true)`. if you instead know that you may want to keep some parts of the old file, call `.truncate(false)`");
            },
        );
    }
}
