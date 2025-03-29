use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_lint::{EarlyContext, EarlyLintPass, Level, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::{FileName, SourceFile, Span, SyntaxContext};
use std::ffi::OsStr;
use std::path::{Component, Path};

declare_clippy_lint! {
    /// ### What it does
    /// Checks that module layout uses only self named module files; bans `mod.rs` files.
    ///
    /// ### Why restrict this?
    /// Having multiple module layout styles in a project can be confusing.
    ///
    /// ### Example
    /// ```text
    /// src/
    ///   stuff/
    ///     stuff_files.rs
    ///     mod.rs
    ///   lib.rs
    /// ```
    /// Use instead:
    /// ```text
    /// src/
    ///   stuff/
    ///     stuff_files.rs
    ///   stuff.rs
    ///   lib.rs
    /// ```
    #[clippy::version = "1.57.0"]
    pub MOD_MODULE_FILES,
    restriction,
    "checks that module layout is consistent"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks that module layout uses only `mod.rs` files.
    ///
    /// ### Why restrict this?
    /// Having multiple module layout styles in a project can be confusing.
    ///
    /// ### Example
    /// ```text
    /// src/
    ///   stuff/
    ///     stuff_files.rs
    ///   stuff.rs
    ///   lib.rs
    /// ```
    /// Use instead:
    /// ```text
    /// src/
    ///   stuff/
    ///     stuff_files.rs
    ///     mod.rs
    ///   lib.rs
    /// ```

    #[clippy::version = "1.57.0"]
    pub SELF_NAMED_MODULE_FILES,
    restriction,
    "checks that module layout is consistent"
}

pub struct ModStyle;

impl_lint_pass!(ModStyle => [MOD_MODULE_FILES, SELF_NAMED_MODULE_FILES]);

impl EarlyLintPass for ModStyle {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, _: &ast::Crate) {
        if cx.builder.lint_level(MOD_MODULE_FILES).level == Level::Allow
            && cx.builder.lint_level(SELF_NAMED_MODULE_FILES).level == Level::Allow
        {
            return;
        }

        let files = cx.sess().source_map().files();

        let Some(trim_to_src) = cx.sess().opts.working_dir.local_path() else {
            return;
        };

        // `folder_segments` is all unique folder path segments `path/to/foo.rs` gives
        // `[path, to]` but not foo
        let mut folder_segments = FxIndexSet::default();
        // `mod_folders` is all the unique folder names that contain a mod.rs file
        let mut mod_folders = FxHashSet::default();
        // `file_map` maps file names to the full path including the file name
        // `{ foo => path/to/foo.rs, .. }
        let mut file_map = FxHashMap::default();
        for file in files.iter() {
            if let FileName::Real(name) = &file.name
                && let Some(lp) = name.local_path()
                && file.cnum == LOCAL_CRATE
            {
                // [#8887](https://github.com/rust-lang/rust-clippy/issues/8887)
                // Only check files in the current crate.
                // Fix false positive that crate dependency in workspace sub directory
                // is checked unintentionally.
                let path = if lp.is_relative() {
                    lp
                } else if let Ok(relative) = lp.strip_prefix(trim_to_src) {
                    relative
                } else {
                    continue;
                };

                if let Some(stem) = path.file_stem() {
                    file_map.insert(stem, (file, path));
                }
                process_paths_for_mod_files(path, &mut folder_segments, &mut mod_folders);
                check_self_named_mod_exists(cx, path, file);
            }
        }

        for folder in &folder_segments {
            if !mod_folders.contains(folder) {
                if let Some((file, path)) = file_map.get(folder) {
                    span_lint_and_then(
                        cx,
                        SELF_NAMED_MODULE_FILES,
                        Span::new(file.start_pos, file.start_pos, SyntaxContext::root(), None),
                        format!("`mod.rs` files are required, found `{}`", path.display()),
                        |diag| {
                            let mut correct = path.to_path_buf();
                            correct.pop();
                            correct.push(folder);
                            correct.push("mod.rs");
                            diag.help(format!("move `{}` to `{}`", path.display(), correct.display(),));
                        },
                    );
                }
            }
        }
    }
}

/// For each `path` we add each folder component to `folder_segments` and if the file name
/// is `mod.rs` we add it's parent folder to `mod_folders`.
fn process_paths_for_mod_files<'a>(
    path: &'a Path,
    folder_segments: &mut FxIndexSet<&'a OsStr>,
    mod_folders: &mut FxHashSet<&'a OsStr>,
) {
    let mut comp = path.components().rev().peekable();
    let _: Option<_> = comp.next();
    if path.ends_with("mod.rs") {
        mod_folders.insert(comp.peek().map(|c| c.as_os_str()).unwrap_or_default());
    }
    let folders = comp.filter_map(|c| if let Component::Normal(s) = c { Some(s) } else { None });
    folder_segments.extend(folders);
}

/// Checks every path for the presence of `mod.rs` files and emits the lint if found.
/// We should not emit a lint for test modules in the presence of `mod.rs`.
/// Using `mod.rs` in integration tests is a [common pattern](https://doc.rust-lang.org/book/ch11-03-test-organization.html#submodules-in-integration-test)
/// for code-sharing between tests.
fn check_self_named_mod_exists(cx: &EarlyContext<'_>, path: &Path, file: &SourceFile) {
    if path.ends_with("mod.rs") && !path.starts_with("tests") {
        span_lint_and_then(
            cx,
            MOD_MODULE_FILES,
            Span::new(file.start_pos, file.start_pos, SyntaxContext::root(), None),
            format!("`mod.rs` files are not allowed, found `{}`", path.display()),
            |diag| {
                let mut mod_file = path.to_path_buf();
                mod_file.pop();
                mod_file.set_extension("rs");

                diag.help(format!("move `{}` to `{}`", path.display(), mod_file.display()));
            },
        );
    }
}
