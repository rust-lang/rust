//! This module contains paths to types and functions Clippy needs to know
//! about.
//!
//! Whenever possible, please consider diagnostic items over hardcoded paths.
//! See <https://github.com/rust-lang/rust-clippy/issues/5393> for more information.

// Paths inside rustc
pub const APPLICABILITY: [&str; 2] = ["rustc_lint_defs", "Applicability"];
pub const APPLICABILITY_VALUES: [[&str; 3]; 4] = [
    ["rustc_lint_defs", "Applicability", "Unspecified"],
    ["rustc_lint_defs", "Applicability", "HasPlaceholders"],
    ["rustc_lint_defs", "Applicability", "MaybeIncorrect"],
    ["rustc_lint_defs", "Applicability", "MachineApplicable"],
];
pub const DIAG: [&str; 2] = ["rustc_errors", "Diag"];
pub const EARLY_CONTEXT: [&str; 2] = ["rustc_lint", "EarlyContext"];
pub const EARLY_LINT_PASS: [&str; 3] = ["rustc_lint", "passes", "EarlyLintPass"];
pub const IDENT: [&str; 3] = ["rustc_span", "symbol", "Ident"];
pub const IDENT_AS_STR: [&str; 4] = ["rustc_span", "symbol", "Ident", "as_str"];
pub const KW_MODULE: [&str; 3] = ["rustc_span", "symbol", "kw"];
pub const LATE_CONTEXT: [&str; 2] = ["rustc_lint", "LateContext"];
pub const LINT: [&str; 2] = ["rustc_lint_defs", "Lint"];
pub const SYMBOL: [&str; 3] = ["rustc_span", "symbol", "Symbol"];
pub const SYMBOL_AS_STR: [&str; 4] = ["rustc_span", "symbol", "Symbol", "as_str"];
pub const SYMBOL_TO_IDENT_STRING: [&str; 4] = ["rustc_span", "symbol", "Symbol", "to_ident_string"];
pub const SYM_MODULE: [&str; 3] = ["rustc_span", "symbol", "sym"];
pub const SYNTAX_CONTEXT: [&str; 3] = ["rustc_span", "hygiene", "SyntaxContext"];

// Paths in `core`/`alloc`/`std`. This should be avoided and cleaned up by adding diagnostic items.
pub const CHAR_IS_ASCII: [&str; 5] = ["core", "char", "methods", "<impl char>", "is_ascii"];
pub const IO_ERROR_NEW: [&str; 5] = ["std", "io", "error", "Error", "new"];
pub const IO_ERRORKIND_OTHER: [&str; 5] = ["std", "io", "error", "ErrorKind", "Other"];
pub const ALIGN_OF: [&str; 3] = ["core", "mem", "align_of"];

// Paths in clippy itself
pub const MSRV_STACK: [&str; 3] = ["clippy_utils", "msrvs", "MsrvStack"];
pub const CLIPPY_SYM_MODULE: [&str; 2] = ["clippy_utils", "sym"];

// Paths in external crates
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const FUTURES_IO_ASYNCREADEXT: [&str; 3] = ["futures_util", "io", "AsyncReadExt"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const FUTURES_IO_ASYNCWRITEEXT: [&str; 3] = ["futures_util", "io", "AsyncWriteExt"];
pub const ITERTOOLS_NEXT_TUPLE: [&str; 3] = ["itertools", "Itertools", "next_tuple"];
pub const PARKING_LOT_MUTEX_GUARD: [&str; 3] = ["lock_api", "mutex", "MutexGuard"];
pub const PARKING_LOT_RWLOCK_READ_GUARD: [&str; 3] = ["lock_api", "rwlock", "RwLockReadGuard"];
pub const PARKING_LOT_RWLOCK_WRITE_GUARD: [&str; 3] = ["lock_api", "rwlock", "RwLockWriteGuard"];
pub const REGEX_BUILDER_NEW: [&str; 3] = ["regex", "RegexBuilder", "new"];
pub const REGEX_BYTES_BUILDER_NEW: [&str; 4] = ["regex", "bytes", "RegexBuilder", "new"];
pub const REGEX_BYTES_NEW: [&str; 4] = ["regex", "bytes", "Regex", "new"];
pub const REGEX_BYTES_SET_NEW: [&str; 4] = ["regex", "bytes", "RegexSet", "new"];
pub const REGEX_NEW: [&str; 3] = ["regex", "Regex", "new"];
pub const REGEX_SET_NEW: [&str; 3] = ["regex", "RegexSet", "new"];
pub const SERDE_DESERIALIZE: [&str; 3] = ["serde", "de", "Deserialize"];
pub const SERDE_DE_VISITOR: [&str; 3] = ["serde", "de", "Visitor"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_FILE_OPTIONS: [&str; 5] = ["tokio", "fs", "file", "File", "options"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_ASYNCREADEXT: [&str; 5] = ["tokio", "io", "util", "async_read_ext", "AsyncReadExt"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_ASYNCWRITEEXT: [&str; 5] = ["tokio", "io", "util", "async_write_ext", "AsyncWriteExt"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_OPEN_OPTIONS: [&str; 4] = ["tokio", "fs", "open_options", "OpenOptions"];
#[expect(clippy::invalid_paths)] // internal lints do not know about all external crates
pub const TOKIO_IO_OPEN_OPTIONS_NEW: [&str; 5] = ["tokio", "fs", "open_options", "OpenOptions", "new"];
