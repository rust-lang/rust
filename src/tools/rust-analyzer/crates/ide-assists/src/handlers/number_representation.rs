use syntax::{AstToken, ast, ast::Radix};

use crate::{AssistContext, AssistId, Assists, GroupLabel};

const MIN_NUMBER_OF_DIGITS_TO_FORMAT: usize = 5;

// Assist: reformat_number_literal
//
// Adds or removes separators from integer literal.
//
// ```
// const _: i32 = 1012345$0;
// ```
// ->
// ```
// const _: i32 = 1_012_345;
// ```
pub(crate) fn reformat_number_literal(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let literal = ctx.find_node_at_offset::<ast::Literal>()?;
    let literal = match literal.kind() {
        ast::LiteralKind::IntNumber(it) => it,
        _ => return None,
    };

    let text = literal.text();
    if text.contains('_') {
        return remove_separators(acc, literal);
    }

    let (prefix, value, suffix) = literal.split_into_parts();
    if value.len() < MIN_NUMBER_OF_DIGITS_TO_FORMAT {
        return None;
    }

    let radix = literal.radix();
    let mut converted = prefix.to_owned();
    converted.push_str(&add_group_separators(value, group_size(radix)));
    converted.push_str(suffix);

    let group_id = GroupLabel("Reformat number literal".into());
    let label = format!("Convert {literal} to {converted}");
    let range = literal.syntax().text_range();
    acc.add_group(
        &group_id,
        AssistId::refactor_inline("reformat_number_literal"),
        label,
        range,
        |builder| builder.replace(range, converted),
    )
}

fn remove_separators(acc: &mut Assists, literal: ast::IntNumber) -> Option<()> {
    let group_id = GroupLabel("Reformat number literal".into());
    let range = literal.syntax().text_range();
    acc.add_group(
        &group_id,
        AssistId::refactor_inline("reformat_number_literal"),
        "Remove digit separators",
        range,
        |builder| builder.replace(range, literal.text().replace('_', "")),
    )
}

const fn group_size(r: Radix) -> usize {
    match r {
        Radix::Binary => 4,
        Radix::Octal => 3,
        Radix::Decimal => 3,
        Radix::Hexadecimal => 4,
    }
}

fn add_group_separators(s: &str, group_size: usize) -> String {
    let mut chars = Vec::new();
    for (i, ch) in s.chars().filter(|&ch| ch != '_').rev().enumerate() {
        if i > 0 && i % group_size == 0 {
            chars.push('_');
        }
        chars.push(ch);
    }

    chars.into_iter().rev().collect()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist_by_label, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn group_separators() {
        let cases = vec![
            ("", 4, ""),
            ("1", 4, "1"),
            ("12", 4, "12"),
            ("123", 4, "123"),
            ("1234", 4, "1234"),
            ("12345", 4, "1_2345"),
            ("123456", 4, "12_3456"),
            ("1234567", 4, "123_4567"),
            ("12345678", 4, "1234_5678"),
            ("123456789", 4, "1_2345_6789"),
            ("1234567890", 4, "12_3456_7890"),
            ("1_2_3_4_5_6_7_8_9_0_", 4, "12_3456_7890"),
            ("1234567890", 3, "1_234_567_890"),
            ("1234567890", 2, "12_34_56_78_90"),
            ("1234567890", 1, "1_2_3_4_5_6_7_8_9_0"),
        ];

        for case in cases {
            let (input, group_size, expected) = case;
            assert_eq!(add_group_separators(input, group_size), expected)
        }
    }

    #[test]
    fn good_targets() {
        let cases = vec![
            ("const _: i32 = 0b11111$0", "0b11111"),
            ("const _: i32 = 0o77777$0;", "0o77777"),
            ("const _: i32 = 10000$0;", "10000"),
            ("const _: i32 = 0xFFFFF$0;", "0xFFFFF"),
            ("const _: i32 = 10000i32$0;", "10000i32"),
            ("const _: i32 = 0b_10_0i32$0;", "0b_10_0i32"),
        ];

        for case in cases {
            check_assist_target(reformat_number_literal, case.0, case.1);
        }
    }

    #[test]
    fn bad_targets() {
        let cases = vec![
            "const _: i32 = 0b111$0",
            "const _: i32 = 0b1111$0",
            "const _: i32 = 0o77$0;",
            "const _: i32 = 0o777$0;",
            "const _: i32 = 10$0;",
            "const _: i32 = 999$0;",
            "const _: i32 = 0xFF$0;",
            "const _: i32 = 0xFFFF$0;",
        ];

        for case in cases {
            check_assist_not_applicable(reformat_number_literal, case);
        }
    }

    #[test]
    fn labels() {
        let cases = vec![
            ("const _: i32 = 10000$0", "const _: i32 = 10_000", "Convert 10000 to 10_000"),
            (
                "const _: i32 = 0xFF0000$0;",
                "const _: i32 = 0xFF_0000;",
                "Convert 0xFF0000 to 0xFF_0000",
            ),
            (
                "const _: i32 = 0b11111111$0;",
                "const _: i32 = 0b1111_1111;",
                "Convert 0b11111111 to 0b1111_1111",
            ),
            (
                "const _: i32 = 0o377211$0;",
                "const _: i32 = 0o377_211;",
                "Convert 0o377211 to 0o377_211",
            ),
            (
                "const _: i32 = 10000i32$0;",
                "const _: i32 = 10_000i32;",
                "Convert 10000i32 to 10_000i32",
            ),
            ("const _: i32 = 1_0_0_0_i32$0;", "const _: i32 = 1000i32;", "Remove digit separators"),
        ];

        for case in cases {
            let (before, after, label) = case;
            check_assist_by_label(reformat_number_literal, before, after, label);
        }
    }
}
