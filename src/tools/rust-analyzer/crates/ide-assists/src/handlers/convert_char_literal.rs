use syntax::{AstToken, ast};

use crate::{AssistContext, AssistId, Assists, GroupLabel};

// Assist: convert_char_literal
//
// Converts character literals between different representations. Currently supports normal character -> ASCII / Unicode escape.
// ```
// const _: char = 'a'$0;
// ```
// ->
// ```
// const _: char = '\x61';
// ```
pub(crate) fn convert_char_literal(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if !ctx.has_empty_selection() {
        return None;
    }

    let literal = ctx.find_node_at_offset::<ast::Literal>()?;
    let literal = match literal.kind() {
        ast::LiteralKind::Char(it) => it,
        _ => return None,
    };

    let value = literal.value().ok()?;
    let text = literal.syntax().text().to_owned();
    let range = literal.syntax().text_range();
    let group_id = GroupLabel("Convert char representation".into());

    let mut add_assist = |converted: String| {
        // Skip no-op assists (e.g. `'const C: char = '\\x61';'` already matches the ASCII form).
        if converted == text {
            return;
        }
        let label = format!("Convert {text} to {converted}");
        acc.add_group(
            &group_id,
            AssistId::refactor_rewrite("convert_char_literal"),
            label,
            range,
            |builder| builder.replace(range, converted),
        );
    };

    if value.is_ascii() {
        add_assist(format!("'\\x{:02x}'", value as u32));
    }

    add_assist(format!("'\\u{{{:x}}}'", value as u32));

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist_by_label;

    use super::convert_char_literal;

    #[test]
    fn ascii_char_to_ascii_and_unicode() {
        let before = "const _: char = 'a'$0;";
        check_assist_by_label(
            convert_char_literal,
            before,
            "const _: char = '\\x61';",
            "Convert 'a' to '\\x61'",
        );
        check_assist_by_label(
            convert_char_literal,
            before,
            "const _: char = '\\u{61}';",
            "Convert 'a' to '\\u{61}'",
        );
    }

    #[test]
    fn non_ascii_char_only_unicode() {
        check_assist_by_label(
            convert_char_literal,
            "const _: char = '😀'$0;",
            "const _: char = '\\u{1f600}';",
            "Convert '😀' to '\\u{1f600}'",
        );
    }

    #[test]
    fn ascii_escape_can_convert_to_unicode() {
        check_assist_by_label(
            convert_char_literal,
            "const _: char = '\\x61'$0;",
            "const _: char = '\\u{61}';",
            "Convert '\\x61' to '\\u{61}'",
        );
    }
}
