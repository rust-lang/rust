use syntax::{ast, ast::Radix, AstToken};

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
    let literal = ctx.find_node_at_offset::<ast::Literal>()?.as_int_number()?;
    let radix = literal.radix();
    let value = literal.value()?;
    let suffix = literal.suffix();

    let range = literal.syntax().text_range();
    let group_id = GroupLabel("Convert integer base".into());

    for &target_radix in Radix::ALL {
        if target_radix == radix {
            continue;
        }

        let mut converted = match target_radix {
            Radix::Binary => format!("0b{:b}", value),
            Radix::Octal => format!("0o{:o}", value),
            Radix::Decimal => value.to_string(),
            Radix::Hexadecimal => format!("0x{:X}", value),
        };

        let label = format!("Convert {} to {}{}", literal, converted, suffix.unwrap_or_default());

        // Appends the type suffix back into the new literal if it exists.
        if let Some(suffix) = suffix {
            converted.push_str(suffix);
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

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist_by_label, check_assist_not_applicable, check_assist_target};

    use super::*;

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
            "Convert 1_00_0 to 0b1111101000",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750;",
            "Convert 1_00_0 to 0o1750",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8;",
            "Convert 1_00_0 to 0x3E8",
        );
    }

    #[test]
    fn convert_small_decimal_integer_with_underscores() {
        let before = "const _: i32 = 1_0<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010;",
            "Convert 1_0 to 0b1010",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12;",
            "Convert 1_0 to 0o12",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xA;",
            "Convert 1_0 to 0xA",
        );
    }

    #[test]
    fn convert_hexadecimal_integer_with_underscores() {
        let before = "const _: i32 = 0x_F_F<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111;",
            "Convert 0x_F_F to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377;",
            "Convert 0x_F_F to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0x_F_F to 255",
        );
    }

    #[test]
    fn convert_binary_integer_with_underscores() {
        let before = "const _: i32 = 0b1111_1111<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377;",
            "Convert 0b1111_1111 to 0o377",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0b1111_1111 to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFF;",
            "Convert 0b1111_1111 to 0xFF",
        );
    }

    #[test]
    fn convert_octal_integer_with_underscores() {
        let before = "const _: i32 = 0o3_77<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111;",
            "Convert 0o3_77 to 0b11111111",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255;",
            "Convert 0o3_77 to 255",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFF;",
            "Convert 0o3_77 to 0xFF",
        );
    }

    #[test]
    fn convert_decimal_integer_with_suffix() {
        let before = "const _: i32 = 1000i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1111101000i32;",
            "Convert 1000i32 to 0b1111101000i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750i32;",
            "Convert 1000i32 to 0o1750i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8i32;",
            "Convert 1000i32 to 0x3E8i32",
        );
    }

    #[test]
    fn convert_small_decimal_integer_with_suffix() {
        let before = "const _: i32 = 10i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010i32;",
            "Convert 10i32 to 0b1010i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12i32;",
            "Convert 10i32 to 0o12i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xAi32;",
            "Convert 10i32 to 0xAi32",
        );
    }

    #[test]
    fn convert_hexadecimal_integer_with_suffix() {
        let before = "const _: i32 = 0xFFi32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0xFFi32 to 0b11111111i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0xFFi32 to 0o377i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0xFFi32 to 255i32",
        );
    }

    #[test]
    fn convert_binary_integer_with_suffix() {
        let before = "const _: i32 = 0b11111111i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0b11111111i32 to 0o377i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0b11111111i32 to 255i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0b11111111i32 to 0xFFi32",
        );
    }

    #[test]
    fn convert_octal_integer_with_suffix() {
        let before = "const _: i32 = 0o377i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0o377i32 to 0b11111111i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0o377i32 to 255i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0o377i32 to 0xFFi32",
        );
    }

    #[test]
    fn convert_decimal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 1_00_0i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1111101000i32;",
            "Convert 1_00_0i32 to 0b1111101000i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o1750i32;",
            "Convert 1_00_0i32 to 0o1750i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0x3E8i32;",
            "Convert 1_00_0i32 to 0x3E8i32",
        );
    }

    #[test]
    fn convert_small_decimal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 1_0i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b1010i32;",
            "Convert 1_0i32 to 0b1010i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o12i32;",
            "Convert 1_0i32 to 0o12i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xAi32;",
            "Convert 1_0i32 to 0xAi32",
        );
    }

    #[test]
    fn convert_hexadecimal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 0x_F_Fi32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0x_F_Fi32 to 0b11111111i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0x_F_Fi32 to 0o377i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0x_F_Fi32 to 255i32",
        );
    }

    #[test]
    fn convert_binary_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 0b1111_1111i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0o377i32;",
            "Convert 0b1111_1111i32 to 0o377i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0b1111_1111i32 to 255i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0b1111_1111i32 to 0xFFi32",
        );
    }

    #[test]
    fn convert_octal_integer_with_underscores_and_suffix() {
        let before = "const _: i32 = 0o3_77i32<|>;";

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0b11111111i32;",
            "Convert 0o3_77i32 to 0b11111111i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 255i32;",
            "Convert 0o3_77i32 to 255i32",
        );

        check_assist_by_label(
            convert_integer_literal,
            before,
            "const _: i32 = 0xFFi32;",
            "Convert 0o3_77i32 to 0xFFi32",
        );
    }

    #[test]
    fn convert_overflowing_literal() {
        let before = "const _: i32 =
            111111111111111111111111111111111111111111111111111111111111111111111111<|>;";
        check_assist_not_applicable(convert_integer_literal, before);
    }
}
