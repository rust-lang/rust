use std::fmt::{self, Display};
use std::io::{self, Write};
use std::path::Path;

use crate::rustfmt_diff::{DiffLine, Mismatch};

/// The checkstyle header - should be emitted before the output of Rustfmt.
///
/// Note that emitting checkstyle output is not stable and may removed in a
/// future version of Rustfmt.
pub(crate) fn header() -> String {
    let mut xml_heading = String::new();
    xml_heading.push_str(r#"<?xml version="1.0" encoding="utf-8"?>"#);
    xml_heading.push_str("\n");
    xml_heading.push_str(r#"<checkstyle version="4.3">"#);
    xml_heading
}

/// The checkstyle footer - should be emitted after the output of Rustfmt.
///
/// Note that emitting checkstyle output is not stable and may removed in a
/// future version of Rustfmt.
pub(crate) fn footer() -> String {
    "</checkstyle>\n".to_owned()
}

pub(crate) fn output_checkstyle_file<T>(
    mut writer: T,
    filename: &Path,
    diff: Vec<Mismatch>,
) -> Result<(), io::Error>
where
    T: Write,
{
    write!(writer, r#"<file name="{}">"#, filename.display())?;
    for mismatch in diff {
        for line in mismatch.lines {
            // Do nothing with `DiffLine::Context` and `DiffLine::Resulting`.
            if let DiffLine::Expected(message) = line {
                write!(
                    writer,
                    r#"<error line="{}" severity="warning" message="Should be `{}`" />"#,
                    mismatch.line_number,
                    XmlEscaped(&message)
                )?;
            }
        }
    }
    write!(writer, "</file>")?;
    Ok(())
}

/// Convert special characters into XML entities.
/// This is needed for checkstyle output.
struct XmlEscaped<'a>(&'a str);

impl<'a> Display for XmlEscaped<'a> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for char in self.0.chars() {
            match char {
                '<' => write!(formatter, "&lt;"),
                '>' => write!(formatter, "&gt;"),
                '"' => write!(formatter, "&quot;"),
                '\'' => write!(formatter, "&apos;"),
                '&' => write!(formatter, "&amp;"),
                _ => write!(formatter, "{}", char),
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
