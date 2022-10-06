//! Completes environment variables defined by Cargo (https://doc.rust-lang.org/cargo/reference/environment-variables.html)

use syntax::{ast, AstToken, AstNode, TextRange, TextSize};

use crate::{context::CompletionContext, CompletionItem, CompletionItemKind};

use super::Completions;
const CARGO_DEFINED_VARS: &[(&str, &str)] = &[
("CARGO","Path to the cargo binary performing the build"),
("CARGO_MANIFEST_DIR","The directory containing the manifest of your package"),
("CARGO_PKG_VERSION","The full version of your package"),
("CARGO_PKG_VERSION_MAJOR","The major version of your package"),
("CARGO_PKG_VERSION_MINOR","The minor version of your package"),
("CARGO_PKG_VERSION_PATCH","The patch version of your package"),
("CARGO_PKG_VERSION_PRE","The pre-release version of your package"),
("CARGO_PKG_AUTHORS","Colon separated list of authors from the manifest of your package"),
("CARGO_PKG_NAME","The name of your package"),
("CARGO_PKG_DESCRIPTION","The description from the manifest of your package"),
("CARGO_PKG_HOMEPAGE","The home page from the manifest of your package"),
("CARGO_PKG_REPOSITORY","The repository from the manifest of your package"),
("CARGO_PKG_LICENSE","The license from the manifest of your package"),
("CARGO_PKG_LICENSE_FILE","The license file from the manifest of your package"),
("CARGO_PKG_RUST_VERSION","The Rust version from the manifest of your package. Note that this is the minimum Rust version supported by the package, not the current Rust version"),
("CARGO_CRATE_NAME","The name of the crate that is currently being compiled"),
("CARGO_BIN_NAME","The name of the binary that is currently being compiled (if it is a binary). This name does not include any file extension, such as .exe"),
("CARGO_PRIMARY_PACKAGE","This environment variable will be set if the package being built is primary. Primary packages are the ones the user selected on the command-line, either with -p flags or the defaults based on the current directory and the default workspace members. This environment variable will not be set when building dependencies. This is only set when compiling the package (not when running binaries or tests)"),
("CARGO_TARGET_TMPDIR","Only set when building integration test or benchmark code. This is a path to a directory inside the target directory where integration tests or benchmarks are free to put any data needed by the tests/benches. Cargo initially creates this directory but doesn't manage its content in any way, this is the responsibility of the test code")
];

pub(crate) fn complete_cargo_env_vars(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    original: &ast::String
) {
    if !is_env_macro(original) {
        return;
    }

    let start = ctx.original_token.text_range().start() + TextSize::from(1);
    let cursor = ctx.position.offset;

    CompletionItem::new(CompletionItemKind::Binding, TextRange::new(start, cursor), "CARGO").add_to(acc);
}

fn is_env_macro(string: &ast::String) -> bool {
    //todo: replace copypaste from format_string with separate function
    (|| {
        let macro_call = string.syntax().parent_ancestors().find_map(ast::MacroCall::cast)?;
        let name = macro_call.path()?.segment()?.name_ref()?;

        if !matches!(name.text().as_str(), 
        "env" | "option_env") {
            return None;
        }


        Some(())
    })()
    .is_some()
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use crate::tests::{check_edit};

    #[test]
    fn completes_env_variables() {
        check_edit("CARGO", 
        r#"
            fn main() {
                let foo = env!("CA$0);
            }
        "#
        ,r#"
            fn main() {
                let foo = env!("CARGO);
            }
        "#)
    }
}