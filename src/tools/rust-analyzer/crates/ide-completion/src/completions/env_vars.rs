//! Completes environment variables defined by Cargo
//! (<https://doc.rust-lang.org/cargo/reference/environment-variables.html>)
use ide_db::syntax_helpers::node_ext::macro_call_for_string_token;
use syntax::{
    AstToken,
    ast::{self, IsString},
};

use crate::{
    CompletionItem, CompletionItemKind, completions::Completions, context::CompletionContext,
};

const CARGO_DEFINED_VARS: &[(&str, &str)] = &[
    ("CARGO", "Path to the cargo binary performing the build"),
    ("CARGO_MANIFEST_DIR", "The directory containing the manifest of your package"),
    ("CARGO_MANIFEST_PATH", "The path to the manifest of your package"),
    ("CARGO_PKG_VERSION", "The full version of your package"),
    ("CARGO_PKG_VERSION_MAJOR", "The major version of your package"),
    ("CARGO_PKG_VERSION_MINOR", "The minor version of your package"),
    ("CARGO_PKG_VERSION_PATCH", "The patch version of your package"),
    ("CARGO_PKG_VERSION_PRE", "The pre-release version of your package"),
    ("CARGO_PKG_AUTHORS", "Colon separated list of authors from the manifest of your package"),
    ("CARGO_PKG_NAME", "The name of your package"),
    ("CARGO_PKG_DESCRIPTION", "The description from the manifest of your package"),
    ("CARGO_PKG_HOMEPAGE", "The home page from the manifest of your package"),
    ("CARGO_PKG_REPOSITORY", "The repository from the manifest of your package"),
    ("CARGO_PKG_LICENSE", "The license from the manifest of your package"),
    ("CARGO_PKG_LICENSE_FILE", "The license file from the manifest of your package"),
    (
        "CARGO_PKG_RUST_VERSION",
        "The Rust version from the manifest of your package. Note that this is the minimum Rust version supported by the package, not the current Rust version",
    ),
    ("CARGO_CRATE_NAME", "The name of the crate that is currently being compiled"),
    (
        "CARGO_BIN_NAME",
        "The name of the binary that is currently being compiled (if it is a binary). This name does not include any file extension, such as .exe",
    ),
    (
        "CARGO_PRIMARY_PACKAGE",
        "This environment variable will be set if the package being built is primary. Primary packages are the ones the user selected on the command-line, either with -p flags or the defaults based on the current directory and the default workspace members. This environment variable will not be set when building dependencies. This is only set when compiling the package (not when running binaries or tests)",
    ),
    (
        "CARGO_TARGET_TMPDIR",
        "Only set when building integration test or benchmark code. This is a path to a directory inside the target directory where integration tests or benchmarks are free to put any data needed by the tests/benches. Cargo initially creates this directory but doesn't manage its content in any way, this is the responsibility of the test code",
    ),
];

pub(crate) fn complete_cargo_env_vars(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    original: &ast::String,
    expanded: &ast::String,
) -> Option<()> {
    let is_in_env_expansion = ctx
        .sema
        .hir_file_for(&expanded.syntax().parent()?)
        .macro_file()
        .is_some_and(|it| it.is_env_or_option_env(ctx.sema.db));
    if !is_in_env_expansion {
        let call = macro_call_for_string_token(expanded)?;
        let makro = ctx.sema.resolve_macro_call(&call)?;
        // We won't map into `option_env` as that generates `None` for non-existent env vars
        // so fall back to this lookup
        if !makro.is_env_or_option_env(ctx.sema.db) {
            return None;
        }
    }
    let range = original.text_range_between_quotes()?;

    CARGO_DEFINED_VARS.iter().for_each(|&(var, detail)| {
        let mut item = CompletionItem::new(CompletionItemKind::Keyword, range, var, ctx.edition);
        item.detail(detail);
        item.add_to(acc, ctx.db);
    });

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_edit, completion_list};

    #[test]
    fn completes_env_variable_in_env() {
        check_edit(
            "CARGO_BIN_NAME",
            r#"
//- minicore: env
fn main() {
    let foo = env!("CAR$0");
}
        "#,
            r#"
fn main() {
    let foo = env!("CARGO_BIN_NAME");
}
        "#,
        );
    }

    #[test]
    fn completes_env_variable_in_option_env() {
        check_edit(
            "CARGO_BIN_NAME",
            r#"
//- minicore: env
fn main() {
    let foo = option_env!("CAR$0");
}
        "#,
            r#"
fn main() {
    let foo = option_env!("CARGO_BIN_NAME");
}
        "#,
        );
    }

    #[test]
    fn doesnt_complete_in_random_strings() {
        let fixture = r#"
            fn main() {
                let foo = "CA$0";
            }
        "#;

        let completions = completion_list(fixture);
        assert!(completions.is_empty(), "Completions weren't empty: {completions}");
    }

    #[test]
    fn doesnt_complete_in_random_macro() {
        let fixture = r#"
            macro_rules! bar {
                ($($arg:tt)*) => { 0 }
            }

            fn main() {
                let foo = bar!("CA$0");

            }
        "#;

        let completions = completion_list(fixture);
        assert!(completions.is_empty(), "Completions weren't empty: {completions}");
    }

    #[test]
    fn doesnt_complete_for_shadowed_macro() {
        let fixture = r#"
            macro_rules! env {
                ($var:literal) => { 0 }
            }

            fn main() {
                let foo = env!("CA$0");
            }
        "#;

        let completions = completion_list(fixture);
        assert!(completions.is_empty(), "Completions weren't empty: {completions}")
    }
}
