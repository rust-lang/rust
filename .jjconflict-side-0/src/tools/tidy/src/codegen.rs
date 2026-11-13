//! Tidy check for codegen backend TODO policy.

use std::ffi::OsStr;
use std::path::Path;

use crate::diagnostics::{CheckId, TidyCtx};
use crate::walk::walk;

fn is_codegen_repo_path(path: &Path) -> bool {
    const CODEGEN_REPO_PATHS: &[&str] =
        &["compiler/rustc_codegen_cranelift", "compiler/rustc_codegen_gcc"];

    CODEGEN_REPO_PATHS.iter().any(|repo| path.ancestors().any(|p| p.ends_with(Path::new(repo))))
}

fn has_supported_extension(path: &Path) -> bool {
    const EXTENSIONS: &[&str] =
        &["rs", "py", "js", "sh", "c", "cpp", "h", "md", "css", "ftl", "toml", "yml", "yaml"];
    path.extension().is_some_and(|ext| EXTENSIONS.iter().any(|e| ext == OsStr::new(e)))
}

pub fn check(path: &Path, tidy_ctx: TidyCtx) {
    let mut check = tidy_ctx.start_check(CheckId::new("codegen").path(path));

    fn skip(path: &Path, is_dir: bool) -> bool {
        if path.file_name().is_some_and(|name| name.to_string_lossy().starts_with(".#")) {
            // Editor temp file.
            return true;
        }

        if is_dir {
            return false;
        }

        // Scan only selected text file types.
        !has_supported_extension(path)
    }

    walk(path, skip, &mut |entry, contents| {
        let file = entry.path();

        if !is_codegen_repo_path(file) {
            return;
        }

        for (i, line) in contents.split('\n').enumerate() {
            let trimmed = line.trim();

            // TODO policy for codegen-only trees.
            if trimmed.contains("TODO") {
                check.error(format!(
                    "{}:{}: TODO is used for tasks that should be done before merging a PR; \
                     if you want to leave a message in the codebase use FIXME",
                    file.display(),
                    i + 1
                ));
            }
        }
    });
}
