use clippy_utils::paths::{PathLookup, PathNS};
use clippy_utils::{sym, type_path, value_path};

// Paths inside rustc
pub static EARLY_LINT_PASS: PathLookup = type_path!(rustc_lint::passes::EarlyLintPass);
pub static KW_MODULE: PathLookup = type_path!(rustc_span::symbol::kw);
pub static LINT: PathLookup = type_path!(rustc_lint_defs::Lint);
pub static SYMBOL: PathLookup = type_path!(rustc_span::symbol::Symbol);
pub static SYMBOL_AS_STR: PathLookup = value_path!(rustc_span::symbol::Symbol::as_str);
pub static SYM_MODULE: PathLookup = type_path!(rustc_span::symbol::sym);
pub static SYNTAX_CONTEXT: PathLookup = type_path!(rustc_span::hygiene::SyntaxContext);

// Paths in clippy itself
pub static CLIPPY_SYM_MODULE: PathLookup = type_path!(clippy_utils::sym);
pub static MSRV_STACK: PathLookup = type_path!(clippy_utils::msrvs::MsrvStack);
pub static PATH_LOOKUP_NEW: PathLookup = value_path!(clippy_utils::paths::PathLookup::new);
pub static SPAN_LINT_AND_THEN: PathLookup = value_path!(clippy_utils::diagnostics::span_lint_and_then);
