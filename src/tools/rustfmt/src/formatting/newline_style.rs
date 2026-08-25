use crate::NewlineStyle;

/// Apply this newline style to the formatted text. When the style is set
/// to `Auto`, the `raw_input_text` is used to detect the existing line
/// endings.
///
/// If the style is set to `Auto` and `raw_input_text` contains no
/// newlines, the `Native` style will be used.
pub(crate) fn apply_newline_style(
    newline_style: NewlineStyle,
    formatted_text: &mut String,
    raw_input_text: &str,
) {
    *formatted_text = match effective_newline_style(newline_style, raw_input_text) {
        EffectiveNewlineStyle::Windows => convert_to_windows_newlines(formatted_text),
        EffectiveNewlineStyle::Unix => convert_to_unix_newlines(formatted_text),
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum EffectiveNewlineStyle {
    Windows,
    Unix,
}

fn effective_newline_style(
    newline_style: NewlineStyle,
    raw_input_text: &str,
) -> EffectiveNewlineStyle {
    match newline_style {
        NewlineStyle::Auto => auto_detect_newline_style(raw_input_text),
        NewlineStyle::Native => native_newline_style(),
        NewlineStyle::Windows => EffectiveNewlineStyle::Windows,
        NewlineStyle::Unix => EffectiveNewlineStyle::Unix,
    }
}

const LINE_FEED: char = '\n';
const CARRIAGE_RETURN: char = '\r';
const WINDOWS_NEWLINE: &str = "\r\n";
const UNIX_NEWLINE: &str = "\n";

fn auto_detect_newline_style(raw_input_text: &str) -> EffectiveNewlineStyle {
    let first_line_feed_pos = raw_input_text.chars().position(|ch| ch == LINE_FEED);
    match first_line_feed_pos {
        Some(first_line_feed_pos) => {
            let char_before_line_feed_pos = first_line_feed_pos.saturating_sub(1);
            let char_before_line_feed = raw_input_text.chars().nth(char_before_line_feed_pos);
            match char_before_line_feed {
                Some(CARRIAGE_RETURN) => EffectiveNewlineStyle::Windows,
                _ => EffectiveNewlineStyle::Unix,
            }
        }
        None => native_newline_style(),
    }
}

fn native_newline_style() -> EffectiveNewlineStyle {
    if cfg!(windows) {
        EffectiveNewlineStyle::Windows
    } else {
        EffectiveNewlineStyle::Unix
    }
}

fn convert_to_windows_newlines(formatted_text: &String) -> String {
    let mut transformed = String::with_capacity(2 * formatted_text.capacity());
    let mut chars = formatted_text.chars().peekable();
    while let Some(current_char) = chars.next() {
        let next_char = chars.peek();
        match current_char {
            LINE_FEED => transformed.push_str(WINDOWS_NEWLINE),
            CARRIAGE_RETURN if next_char == Some(&LINE_FEED) => {}
            current_char => transformed.push(current_char),
        }
    }
    transformed
}

fn convert_to_unix_newlines(formatted_text: &str) -> String {
    formatted_text.replace(WINDOWS_NEWLINE, UNIX_NEWLINE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detects_unix_newlines() {
        assert_eq!(
            EffectiveNewlineStyle::Unix,
            auto_detect_newline_style("One\nTwo\nThree")
        );
    }

    #[test]
    fn auto_detects_windows_newlines() {
        assert_eq!(
            EffectiveNewlineStyle::Windows,
            auto_detect_newline_style("One\r\nTwo\r\nThree")
        );
    }

    #[test]
    fn auto_detects_windows_newlines_with_multibyte_char_on_first_line() {
        assert_eq!(
            EffectiveNewlineStyle::Windows,
            auto_detect_newline_style("A 🎢 of a first line\r\nTwo\r\nThree")
        );
    }

    #[test]
    fn falls_back_to_native_newlines_if_no_newlines_are_found() {
        let expected_newline_style = if cfg!(windows) {
            EffectiveNewlineStyle::Windows
        } else {
            EffectiveNewlineStyle::Unix
        };
        assert_eq!(
            expected_newline_style,
            auto_detect_newline_style("One Two Three")
        );
    }

    #[test]
    fn auto_detects_and_applies_unix_newlines() {
        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One\nTwo\nThree";

        let mut out = String::from(formatted_text);
        apply_newline_style(NewlineStyle::Auto, &mut out, raw_input_text);
        assert_eq!("One\nTwo\nThree", &out, "auto should detect 'lf'");
    }

    #[test]
    fn auto_detects_and_applies_windows_newlines() {
        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One\r\nTwo\r\nThree";

        let mut out = String::from(formatted_text);
        apply_newline_style(NewlineStyle::Auto, &mut out, raw_input_text);
        assert_eq!("One\r\nTwo\r\nThree", &out, "auto should detect 'crlf'");
    }

    #[test]
    fn auto_detects_and_applies_native_newlines() {
        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One Two Three";

        let mut out = String::from(formatted_text);
        apply_newline_style(NewlineStyle::Auto, &mut out, raw_input_text);

        if cfg!(windows) {
            assert_eq!(
                "One\r\nTwo\r\nThree", &out,
                "auto-native-windows should detect 'crlf'"
            );
        } else {
            assert_eq!(
                "One\nTwo\nThree", &out,
                "auto-native-unix should detect 'lf'"
            );
        }
    }

    #[test]
    fn applies_unix_newlines() {
        test_newlines_are_applied_correctly(
            "One\r\nTwo\nThree",
            "One\nTwo\nThree",
            NewlineStyle::Unix,
        );
    }

    #[test]
    fn applying_unix_newlines_changes_nothing_for_unix_newlines() {
        let formatted_text = "One\nTwo\nThree";
        test_newlines_are_applied_correctly(formatted_text, formatted_text, NewlineStyle::Unix);
    }

    #[test]
    fn applies_unix_newlines_to_string_with_unix_and_windows_newlines() {
        test_newlines_are_applied_correctly(
            "One\r\nTwo\r\nThree\nFour",
            "One\nTwo\nThree\nFour",
            NewlineStyle::Unix,
        );
    }

    #[test]
    fn applies_windows_newlines_to_string_with_unix_and_windows_newlines() {
        test_newlines_are_applied_correctly(
            "One\nTwo\nThree\r\nFour",
            "One\r\nTwo\r\nThree\r\nFour",
            NewlineStyle::Windows,
        );
    }

    #[test]
    fn applying_windows_newlines_changes_nothing_for_windows_newlines() {
        let formatted_text = "One\r\nTwo\r\nThree";
        test_newlines_are_applied_correctly(formatted_text, formatted_text, NewlineStyle::Windows);
    }

    #[test]
    fn keeps_carriage_returns_when_applying_windows_newlines_to_str_with_unix_newlines() {
        test_newlines_are_applied_correctly(
            "One\nTwo\nThree\rDrei",
            "One\r\nTwo\r\nThree\rDrei",
            NewlineStyle::Windows,
        );
    }

    #[test]
    fn keeps_carriage_returns_when_applying_unix_newlines_to_str_with_unix_newlines() {
        test_newlines_are_applied_correctly(
            "One\nTwo\nThree\rDrei",
            "One\nTwo\nThree\rDrei",
            NewlineStyle::Unix,
        );
    }

    #[test]
    fn keeps_carriage_returns_when_applying_windows_newlines_to_str_with_windows_newlines() {
        test_newlines_are_applied_correctly(
            "One\r\nTwo\r\nThree\rDrei",
            "One\r\nTwo\r\nThree\rDrei",
            NewlineStyle::Windows,
        );
    }

    #[test]
    fn keeps_carriage_returns_when_applying_unix_newlines_to_str_with_windows_newlines() {
        test_newlines_are_applied_correctly(
            "One\r\nTwo\r\nThree\rDrei",
            "One\nTwo\nThree\rDrei",
            NewlineStyle::Unix,
        );
    }

    fn test_newlines_are_applied_correctly(
        input: &str,
        expected: &str,
        newline_style: NewlineStyle,
    ) {
        let mut out = String::from(input);
        apply_newline_style(newline_style, &mut out, input);
        assert_eq!(expected, &out);
    }
}
