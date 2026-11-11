use clippy_utils::diagnostics::span_lint;
use rustc_ast::PatKind;
use rustc_ast::visit::FnKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_lint::EarlyContext;
use rustc_span::Span;

use super::DUPLICATE_UNDERSCORE_ARGUMENT;

pub(super) fn check(cx: &EarlyContext<'_>, fn_kind: FnKind<'_>) {
    let mut registered_names: FxHashMap<String, Span> = FxHashMap::default();

    for arg in &fn_kind.decl().inputs {
        if let PatKind::Ident(_, ident, None) = arg.pat.kind {
            let arg_name = ident.to_string();

            if let Some(arg_name) = arg_name.strip_prefix('_') {
                if let Some(correspondence) = registered_names.get(arg_name) {
                    span_lint(
                        cx,
                        DUPLICATE_UNDERSCORE_ARGUMENT,
                        *correspondence,
                        format!(
                            "`{arg_name}` already exists, having another argument having almost the same \
                                 name makes code comprehension and documentation more difficult"
                        ),
                    );
                }
            } else {
                registered_names.insert(arg_name, arg.pat.span);
            }
        }
    }
}
