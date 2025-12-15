//! Completes identifiers in format string literals.

use hir::{ModuleDef, ScopeDef};
use ide_db::{SymbolKind, syntax_helpers::format_string::is_format_string};
use itertools::Itertools;
use syntax::{AstToken, TextRange, TextSize, ToSmolStr, ast};

use crate::{CompletionItem, CompletionItemKind, Completions, context::CompletionContext};

/// Complete identifiers in format strings.
pub(crate) fn format_string(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    original: &ast::String,
    expanded: &ast::String,
) {
    if !is_format_string(expanded) {
        return;
    }
    let cursor = ctx.position.offset;
    let lit_start = ctx.original_token.text_range().start();
    let cursor_in_lit = cursor - lit_start;

    let prefix = &original.text()[..cursor_in_lit.into()];
    let Some(brace_offset) = unescaped_brace(prefix) else { return };
    let brace_offset = lit_start + brace_offset + TextSize::of('{');

    let source_range = TextRange::new(brace_offset, cursor);
    ctx.locals.iter().sorted_by_key(|&(k, _)| k.clone()).for_each(|(name, _)| {
        CompletionItem::new(
            CompletionItemKind::Binding,
            source_range,
            name.display_no_db(ctx.edition).to_smolstr(),
            ctx.edition,
        )
        .add_to(acc, ctx.db);
    });
    ctx.scope.process_all_names(&mut |name, scope| {
        if let ScopeDef::ModuleDef(module_def) = scope {
            let symbol_kind = match module_def {
                ModuleDef::Const(..) => SymbolKind::Const,
                ModuleDef::Static(..) => SymbolKind::Static,
                _ => return,
            };

            CompletionItem::new(
                CompletionItemKind::SymbolKind(symbol_kind),
                source_range,
                name.display_no_db(ctx.edition).to_smolstr(),
                ctx.edition,
            )
            .add_to(acc, ctx.db);
        }
    });
}

fn unescaped_brace(prefix: &str) -> Option<TextSize> {
    let is_ident_char = |ch: char| ch.is_alphanumeric() || ch == '_';
    prefix
        .trim_end_matches(is_ident_char)
        .strip_suffix('{')
        .filter(|it| it.chars().rev().take_while(|&ch| ch == '{').count() % 2 == 0)
        .map(|s| TextSize::new(s.len() as u32))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::tests::{check_edit, check_no_kw};

    #[test]
    fn works_when_wrapped() {
        check_no_kw(
            r#"
//- minicore: fmt
macro_rules! print {
    ($($arg:tt)*) => (std::io::_print(format_args!($($arg)*)));
}
fn main() {
    let foobar = 1;
    print!("f$0");
}
"#,
            expect![[]],
        );
    }

    #[test]
    fn no_completion_without_brace() {
        check_no_kw(
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("f$0");
}
"#,
            expect![[]],
        );
    }

    #[test]
    fn no_completion_after_escaped() {
        check_no_kw(
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("{{f$0");
}
"#,
            expect![[]],
        );
        check_no_kw(
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("some text {{{{f$0");
}
"#,
            expect![[]],
        );
    }

    #[test]
    fn completes_unescaped_after_escaped() {
        check_edit(
            "foobar",
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("{{{f$0");
}
"#,
            r#"
fn main() {
    let foobar = 1;
    format_args!("{{{foobar");
}
"#,
        );
        check_edit(
            "foobar",
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("{{{{{f$0");
}
"#,
            r#"
fn main() {
    let foobar = 1;
    format_args!("{{{{{foobar");
}
"#,
        );
        check_edit(
            "foobar",
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("}}{f$0");
}
"#,
            r#"
fn main() {
    let foobar = 1;
    format_args!("}}{foobar");
}
"#,
        );
    }

    #[test]
    fn completes_locals() {
        check_edit(
            "foobar",
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("{f$0");
}
"#,
            r#"
fn main() {
    let foobar = 1;
    format_args!("{foobar");
}
"#,
        );
        check_edit(
            "foobar",
            r#"
//- minicore: fmt
fn main() {
    let foobar = 1;
    format_args!("{$0");
}
"#,
            r#"
fn main() {
    let foobar = 1;
    format_args!("{foobar");
}
"#,
        );
    }

    #[test]
    fn completes_constants() {
        check_edit(
            "FOOBAR",
            r#"
//- minicore: fmt
fn main() {
    const FOOBAR: usize = 42;
    format_args!("{f$0");
}
"#,
            r#"
fn main() {
    const FOOBAR: usize = 42;
    format_args!("{FOOBAR");
}
"#,
        );

        check_edit(
            "FOOBAR",
            r#"
//- minicore: fmt
fn main() {
    const FOOBAR: usize = 42;
    format_args!("{$0");
}
"#,
            r#"
fn main() {
    const FOOBAR: usize = 42;
    format_args!("{FOOBAR");
}
"#,
        );
    }

    #[test]
    fn completes_static_constants() {
        check_edit(
            "FOOBAR",
            r#"
//- minicore: fmt
fn main() {
    static FOOBAR: usize = 42;
    format_args!("{f$0");
}
"#,
            r#"
fn main() {
    static FOOBAR: usize = 42;
    format_args!("{FOOBAR");
}
"#,
        );

        check_edit(
            "FOOBAR",
            r#"
//- minicore: fmt
fn main() {
    static FOOBAR: usize = 42;
    format_args!("{$0");
}
"#,
            r#"
fn main() {
    static FOOBAR: usize = 42;
    format_args!("{FOOBAR");
}
"#,
        );
    }
}
