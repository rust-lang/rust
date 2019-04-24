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
    use crate::NewlineStyle::*;
    let mut style = newline_style;
    if style == Auto {
        style = auto_detect_newline_style(raw_input_text);
    }
    if style == Native {
        style = native();
    }
    match style {
        Windows => {
            let mut transformed = String::with_capacity(2 * formatted_text.capacity());
            for c in formatted_text.chars() {
                match c {
                    '\n' => transformed.push_str("\r\n"),
                    '\r' => continue,
                    c => transformed.push(c),
                }
            }
            *formatted_text = transformed;
        }
        Unix => return,
        Native => unreachable!("NewlineStyle::Native"),
        Auto => unreachable!("NewlineStyle::Auto"),
    }
}

fn auto_detect_newline_style(raw_input_text: &str) -> NewlineStyle {
    if let Some(pos) = raw_input_text.find('\n') {
        let pos = pos.saturating_sub(1);
        if let Some('\r') = raw_input_text.chars().nth(pos) {
            NewlineStyle::Windows
        } else {
            NewlineStyle::Unix
        }
    } else {
        NewlineStyle::Native
    }
}

fn native() -> NewlineStyle {
    if cfg!(windows) {
        NewlineStyle::Windows
    } else {
        NewlineStyle::Unix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newline_style_auto_detect() {
        let lf = "One\nTwo\nThree";
        let crlf = "One\r\nTwo\r\nThree";
        let none = "One Two Three";

        assert_eq!(NewlineStyle::Unix, auto_detect_newline_style(lf));
        assert_eq!(NewlineStyle::Windows, auto_detect_newline_style(crlf));
        assert_eq!(NewlineStyle::Native, auto_detect_newline_style(none));
    }

    #[test]
    fn test_newline_style_auto_apply() {
        let auto = NewlineStyle::Auto;

        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One\nTwo\nThree";

        let mut out = String::from(formatted_text);
        apply_newline_style(auto, &mut out, raw_input_text);
        assert_eq!("One\nTwo\nThree", &out, "auto should detect 'lf'");

        let formatted_text = "One\nTwo\nThree";
        let raw_input_text = "One\r\nTwo\r\nThree";

        let mut out = String::from(formatted_text);
        apply_newline_style(auto, &mut out, raw_input_text);
        assert_eq!("One\r\nTwo\r\nThree", &out, "auto should detect 'crlf'");

        #[cfg(not(windows))]
        {
            let formatted_text = "One\nTwo\nThree";
            let raw_input_text = "One Two Three";

            let mut out = String::from(formatted_text);
            apply_newline_style(auto, &mut out, raw_input_text);
            assert_eq!(
                "One\nTwo\nThree", &out,
                "auto-native-unix should detect 'lf'"
            );
        }

        #[cfg(windows)]
        {
            let formatted_text = "One\nTwo\nThree";
            let raw_input_text = "One Two Three";

            let mut out = String::from(formatted_text);
            apply_newline_style(auto, &mut out, raw_input_text);
            assert_eq!(
                "One\r\nTwo\r\nThree", &out,
                "auto-native-windows should detect 'crlf'"
            );
        }
    }
}
