use syntax::{ast, AstNode, SmolStr};

use crate::{AssistContext, AssistId, AssistKind, Assists, GroupLabel};

// Assist: convert_integer_literal
//
// Converts the base of integer literals to other bases.
//
// ```
// const _: i32 = 10<|>;
// ```
// ->
// ```
// const _: i32 = 0b1010;
// ```
pub(crate) fn convert_integer_literal(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let literal = ctx.find_node_at_offset::<ast::Literal>()?;
    let range = literal.syntax().text_range();
    let group_id = GroupLabel("Convert integer base".into());

    let suffix = match literal.kind() {
        ast::LiteralKind::IntNumber { suffix } => suffix,
        _ => return None,
    };
    let suffix_len = suffix.as_ref().map(|s| s.len()).unwrap_or(0);
    let raw_literal_text = literal.syntax().to_string();

    // Gets the literal's text without the type suffix and without underscores.
    let literal_text = raw_literal_text
        .chars()
        .take(raw_literal_text.len() - suffix_len)
        .filter(|c| *c != '_')
        .collect::<SmolStr>();
    let literal_base = IntegerLiteralBase::identify(&literal_text)?;

    for base in IntegerLiteralBase::bases() {
        if *base == literal_base {
            continue;
        }

        let mut converted = literal_base.convert(&literal_text, base);

        let label = if let Some(suffix) = &suffix {
            format!("Convert {} ({}) to {}", &literal_text, suffix, &converted)
        } else {
            format!("Convert {} to {}", &literal_text, &converted)
        };

        // Appends the type suffix back into the new literal if it exists.
        if let Some(suffix) = &suffix {
            converted.push_str(&suffix);
        }

        acc.add_group(
            &group_id,
            AssistId("convert_integer_literal", AssistKind::RefactorInline),
            label,
            range,
            |builder| builder.replace(range, converted),
        );
    }

    Some(())
}

#[derive(Debug, PartialEq, Eq)]
enum IntegerLiteralBase {
    Binary,
    Octal,
    Decimal,
    Hexadecimal,
}

impl IntegerLiteralBase {
    fn identify(literal_text: &str) -> Option<Self> {
        // We cannot express a literal in anything other than decimal in under 3 characters, so we return here if possible.
        if literal_text.len() < 3 && literal_text.chars().all(|c| c.is_digit(10)) {
            return Some(Self::Decimal);
        }

        let base = match &literal_text[..2] {
            "0b" => Self::Binary,
            "0o" => Self::Octal,
            "0x" => Self::Hexadecimal,
            _ => Self::Decimal,
        };

        // Checks that all characters after the base prefix are all valid digits for that base.
        if literal_text[base.prefix_len()..].chars().all(|c| c.is_digit(base.base())) {
            Some(base)
        } else {
            None
        }
    }

    fn convert(&self, literal_text: &str, to: &IntegerLiteralBase) -> String {
        let digits = &literal_text[self.prefix_len()..];
        let value = u128::from_str_radix(digits, self.base()).unwrap();

        match to {
            Self::Binary => format!("0b{:b}", value),
            Self::Octal => format!("0o{:o}", value),
            Self::Decimal => value.to_string(),
            Self::Hexadecimal => format!("0x{:X}", value),
        }
    }

    const fn base(&self) -> u32 {
        match self {
            Self::Binary => 2,
            Self::Octal => 8,
            Self::Decimal => 10,
            Self::Hexadecimal => 16,
        }
    }

    const fn prefix_len(&self) -> usize {
        match self {
            Self::Decimal => 0,
            _ => 2,
        }
    }

    const fn bases() -> &'static [IntegerLiteralBase] {
        &[
            IntegerLiteralBase::Binary,
            IntegerLiteralBase::Octal,
            IntegerLiteralBase::Decimal,
            IntegerLiteralBase::Hexadecimal,
        ]
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::tests::{check_assist_by_label, check_assist_target};

    #[test]
    fn binary_target() {
        check_assist_target(convert_integer_literal, "const _: i32 = 0b1010<|>;", "0b1010");
    }

    #[test]
    fn octal_target() {
        check_assist_target(convert_integer_literal, "const _: i32 = 0o12<|>;", "0o12");
    }

    #[test]
    fn decimal_target() {
        check_assist_target(convert_integer_literal, "const _: i32 = 10<|>;", "10");
    }

    #[test]
    fn hexadecimal_target() {
        check_assist_target(convert_integer_literal, "const _: i32 = 0xA<|>;", "0xA");
    }

    #[test]
    fn binary_target_with_underscores() {
        check_assist_target(convert_integer_literal, "const _: i32 = 0b10_10<|>;", "0b10_10");
    }

    #[test]
    fn octal_target_with_underscores() {
        check_assist_target(convert_integer_literal, "const _: i32 = 0o1_2<|>;", "0o1_2");
    }

    #[test]
    fn decimal_target_with_underscores() {
        check_assist_target(convert_integer_literal, "const _: i32 = 1_0<|>;", "1_0");
    }

    #[test]
    fn hexadecimal_target_with_underscores() {
        check_assist_target(convert_integer_literal, "const _: i32 = 0x_A<|>;", "0x_A");
    }

    #[test]
    fn convert_decimal_integer() {
        let before = "const _: i32 = 1000<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1111101000;",
            "Convert 1000 to 0b1111101000",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750;",
            "Convert 1000 to 0o1750",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8;",
            "Convert 1000 to 0x3E8",
        );
    }

    // Decimal numbers under 3 digits have a special case where they return early because we can't fit a
    // other base's prefix, so we have a separate test for that.
    #[test]
    fn convert_small_decimal_integer() {
        let before = "const _: i32 = 10<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010;",
            "Convert 10 to 0b1010",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12;",
            "Convert 10 to 0o12",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xA;",
            "Convert 10 to 0xA",
        );
    }

    #[test]
    fn convert_hexadecimal_integer() {
        let before = "const _: i32 = 0xFF<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111;",
            "Convert 0xFF to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377;",
            "Convert 0xFF to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0xFF to 255",
        );
    }

    #[test]
    fn convert_binary_integer() {
        let before = "const _: i32 = 0b11111111<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377;",
            "Convert 0b11111111 to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0b11111111 to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFF;",
            "Convert 0b11111111 to 0xFF",
        );
    }

    #[test]
    fn convert_octal_integer() {
        let before = "const _: i32 = 0o377<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111;",
            "Convert 0o377 to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0o377 to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFF;",
            "Convert 0o377 to 0xFF",
        );
    }

    #[test]
    fn convert_decimal_integer_with_underscores() {
        let before = "const _: i32 = 1_00_0<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1111101000;",
            "Convert 1000 to 0b1111101000",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750;",
            "Convert 1000 to 0o1750",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8;",
            "Convert 1000 to 0x3E8",
        );
    }

    #[test]
    fn convert_small_decimal_integer_with_underscores() {
        let before = "const _: i32 = 1_0<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010;",
            "Convert 10 to 0b1010",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12;",
            "Convert 10 to 0o12",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xA;",
            "Convert 10 to 0xA",
        );
    }

    #[test]
    fn convert_hexadecimal_integer_with_underscores() {
        let before = "const _: i32 = 0x_F_F<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111;",
            "Convert 0xFF to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377;",
            "Convert 0xFF to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0xFF to 255",
        );
    }

    #[test]
    fn convert_binary_integer_with_underscores() {
        let before = "const _: i32 = 0b1111_1111<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377;",
            "Convert 0b11111111 to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0b11111111 to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFF;",
            "Convert 0b11111111 to 0xFF",
        );
    }

    #[test]
    fn convert_octal_integer_with_underscores() {
        let before = "const _: i32 = 0o3_77<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111;",
            "Convert 0o377 to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0o377 to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFF;",
            "Convert 0o377 to 0xFF",
        );
    }

    #[test]
    fn convert_decimal_integer_with_suffix() {
        let before = "const _: i32 = 1000i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1111101000i32;",
            "Convert 1000 (i32) to 0b1111101000",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750i32;",
            "Convert 1000 (i32) to 0o1750",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8i32;",
            "Convert 1000 (i32) to 0x3E8",
        );
    }

    #[test]
    fn convert_small_decimal_integer_with_suffix() {
        let before = "const _: i32 = 10i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010i32;",
            "Convert 10 (i32) to 0b1010",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12i32;",
            "Convert 10 (i32) to 0o12",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xAi32;",
            "Convert 10 (i32) to 0xA",
        );
    }

    #[test]
    fn convert_hexadecimal_integer_with_suffix() {
        let before = "const _: i32 = 0xFFi32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0xFF (i32) to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0xFF (i32) to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0xFF (i32) to 255",
        );
    }

    #[test]
    fn convert_binary_integer_with_suffix() {
        let before = "const _: i32 = 0b11111111i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0b11111111 (i32) to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0b11111111 (i32) to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0b11111111 (i32) to 0xFF",
        );
    }

    #[test]
    fn convert_octal_integer_with_suffix() {
        let before = "const _: i32 = 0o377i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0o377 (i32) to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0o377 (i32) to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0o377 (i32) to 0xFF",
        );
    }

    #[test]
    fn convert_decimal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 1_00_0i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1111101000i32;",
            "Convert 1000 (i32) to 0b1111101000",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750i32;",
            "Convert 1000 (i32) to 0o1750",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8i32;",
            "Convert 1000 (i32) to 0x3E8",
        );
    }

    #[test]
    fn convert_small_decimal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 1_0i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010i32;",
            "Convert 10 (i32) to 0b1010",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12i32;",
            "Convert 10 (i32) to 0o12",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xAi32;",
            "Convert 10 (i32) to 0xA",
        );
    }

    #[test]
    fn convert_hexadecimal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 0x_F_Fi32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0xFF (i32) to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0xFF (i32) to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0xFF (i32) to 255",
        );
    }

    #[test]
    fn convert_binary_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 0b1111_1111i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0b11111111 (i32) to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0b11111111 (i32) to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0b11111111 (i32) to 0xFF",
        );
    }

    #[test]
    fn convert_octal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 0o3_77i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0o377 (i32) to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0o377 (i32) to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0o377 (i32) to 0xFF",
        );
    }
}
