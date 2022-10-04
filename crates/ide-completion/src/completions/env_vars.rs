//! Completes environment variables defined by Cargo (https://doc.rust-lang.org/cargo/reference/environment-variables.html)

use syntax::{ast, AstToken, AstNode, TextRange, TextSize};

use crate::{context::CompletionContext, CompletionItem, CompletionItemKind};

use super::Completions;

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