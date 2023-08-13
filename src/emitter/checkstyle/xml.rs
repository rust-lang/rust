use std::fmt::{self, Display};

/// Convert special characters into XML entities.
/// This is needed for checkstyle output.
pub(super) struct XmlEscaped<'a>(pub(super) &'a str);

impl<'a> Display for XmlEscaped<'a> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for char in self.0.chars() {
            match char {
                '<' => write!(formatter, "&lt;"),
                '>' => write!(formatter, "&gt;"),
                '"' => write!(formatter, "&quot;"),
                '\'' => write!(formatter, "&apos;"),
                '&' => write!(formatter, "&amp;"),
                _ => write!(formatter, "{char}"),
            }?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_characters_are_escaped() {
        assert_eq!(
            "&lt;&gt;&quot;&apos;&amp;",
            format!("{}", XmlEscaped(r#"<>"'&"#)),
        );
    }

    #[test]
    fn special_characters_are_escaped_in_string_with_other_characters() {
        assert_eq!(
            "The quick brown &quot;🦊&quot; jumps &lt;over&gt; the lazy 🐶",
            format!(
                "{}",
                XmlEscaped(r#"The quick brown "🦊" jumps <over> the lazy 🐶"#)
            ),
        );
    }

    #[test]
    fn other_characters_are_not_escaped() {
        let string = "The quick brown 🦊 jumps over the lazy 🐶";
        assert_eq!(string, format!("{}", XmlEscaped(string)));
    }
}
