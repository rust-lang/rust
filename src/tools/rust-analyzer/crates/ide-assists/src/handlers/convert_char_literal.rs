use syntax::{AstToken, ast};

use crate::{AssistContext, AssistId, Assists, GroupLabel};

// Assist: convert_char_literal
//
// Converts character literals between different representations. Currently supports normal character -> ASCII / Unicode escape.
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
    let text = literal.syntax().text().to_string();
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
