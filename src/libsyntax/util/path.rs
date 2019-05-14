use crate::source_map::SourceMap;
use std::path::PathBuf;
use syntax_pos::{Span, FileName};

/// Resolve a path mentioned inside Rust code.
///
/// This unifies the logic used for resolving `include_X!`, and `#[doc(include)]` file paths.
///
/// Returns an absolute path to the file that `path` refers to.
pub fn resolve(path: impl Into<PathBuf>, span: Span, map: &SourceMap) -> PathBuf {
    let path = path.into();

    // Relative paths are resolved relative to the file in which they are found
    // after macro expansion (that is, they are unhygienic).
    if !path.is_absolute() {
        let callsite = span.source_callsite();
        let mut result = match map.span_to_unmapped_path(callsite) {
            FileName::Real(path) => path,
            FileName::DocTest(path, _) => path,
            other => panic!("cannot resolve relative path in non-file source `{}`", other),
        };
        result.pop();
        result.push(path);
        result
    } else {
        path
    }
}
