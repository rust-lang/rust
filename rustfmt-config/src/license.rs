use regex;

/// Convert the license template into a string which can be turned into a regex.
///
/// The license template could use regex syntax directly, but that would require a lot of manual
/// escaping, which is inconvenient. It is therefore literal by default, with optional regex
/// subparts delimited by `{` and `}`. Additionally:
///
/// - to insert literal `{`, `}` or `\`, escape it with `\`
/// - an empty regex placeholder (`{}`) is shorthand for `{.*?}`
///
/// This function parses this input format and builds a properly escaped *string* representation of
/// the equivalent regular expression. It **does not** however guarantee that the returned string is
/// a syntactically valid regular expression.
///
/// # Examples
///
/// ```
/// # use rustfmt_config::license;
/// assert_eq!(
///     license::parse_template(
///         r"
/// // Copyright {\d+} The \} Rust \\ Project \{ Developers. See the {([A-Z]+)}
/// // file at the top-level directory of this distribution and at
/// // {}.
/// //
/// // Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
/// // http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
/// // <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
/// // option. This file may not be copied, modified, or distributed
/// // except according to those terms.
/// "
///     ).unwrap(),
///     r"^
/// // Copyright \d+ The \} Rust \\ Project \{ Developers\. See the ([A-Z]+)
/// // file at the top\-level directory of this distribution and at
/// // .*?\.
/// //
/// // Licensed under the Apache License, Version 2\.0 <LICENSE\-APACHE or
/// // http://www\.apache\.org/licenses/LICENSE\-2\.0> or the MIT license
/// // <LICENSE\-MIT or http://opensource\.org/licenses/MIT>, at your
/// // option\. This file may not be copied, modified, or distributed
/// // except according to those terms\.
/// "
/// );
/// ```
pub fn parse_template(template: &str) -> Result<String, String> {
    // the template is parsed using a state machine
    enum State {
        Lit,
        LitEsc,
        // the u32 keeps track of brace nesting
        Re(u32),
        ReEsc(u32),
    }

    let mut parsed = String::from("^");
    let mut buffer = String::new();
    let mut state = State::Lit;
    let mut linum = 1;
    // keeps track of last line on which a regex placeholder was started
    let mut open_brace_line = 0;
    for chr in template.chars() {
        if chr == '\n' {
            linum += 1;
        }
        state = match state {
            State::Lit => match chr {
                '{' => {
                    parsed.push_str(&regex::escape(&buffer));
                    buffer.clear();
                    open_brace_line = linum;
                    State::Re(1)
                }
                '}' => return Err(format!("escape or balance closing brace on l. {}", linum)),
                '\\' => State::LitEsc,
                _ => {
                    buffer.push(chr);
                    State::Lit
                }
            },
            State::LitEsc => {
                buffer.push(chr);
                State::Lit
            }
            State::Re(brace_nesting) => {
                match chr {
                    '{' => {
                        buffer.push(chr);
                        State::Re(brace_nesting + 1)
                    }
                    '}' => {
                        match brace_nesting {
                            1 => {
                                // default regex for empty placeholder {}
                                if buffer.is_empty() {
                                    buffer = ".*?".to_string();
                                }
                                parsed.push_str(&buffer);
                                buffer.clear();
                                State::Lit
                            }
                            _ => {
                                buffer.push(chr);
                                State::Re(brace_nesting - 1)
                            }
                        }
                    }
                    '\\' => {
                        buffer.push(chr);
                        State::ReEsc(brace_nesting)
                    }
                    _ => {
                        buffer.push(chr);
                        State::Re(brace_nesting)
                    }
                }
            }
            State::ReEsc(brace_nesting) => {
                buffer.push(chr);
                State::Re(brace_nesting)
            }
        }
    }
    match state {
        State::Re(_) | State::ReEsc(_) => {
            return Err(format!(
                "escape or balance opening brace on l. {}",
                open_brace_line
            ));
        }
        State::LitEsc => return Err(format!("incomplete escape sequence on l. {}", linum)),
        _ => (),
    }
    parsed.push_str(&regex::escape(&buffer));

    Ok(parsed)
}

#[cfg(test)]
mod test {
    use super::parse_template;

    #[test]
    fn test_parse_license_template() {
        assert_eq!(
            parse_template("literal (.*)").unwrap(),
            r"^literal \(\.\*\)"
        );
        assert_eq!(parse_template(r"escaping \}").unwrap(), r"^escaping \}");
        assert!(parse_template("unbalanced } without escape").is_err());
        assert_eq!(
            parse_template(r"{\d+} place{-?}holder{s?}").unwrap(),
            r"^\d+ place-?holders?"
        );
        assert_eq!(parse_template("default {}").unwrap(), "^default .*?");
        assert_eq!(
            parse_template(r"unbalanced nested braces {\{{3}}").unwrap(),
            r"^unbalanced nested braces \{{3}"
        );
        assert_eq!(
            parse_template("parsing error }").unwrap_err(),
            "escape or balance closing brace on l. 1"
        );
        assert_eq!(
            parse_template("parsing error {\nsecond line").unwrap_err(),
            "escape or balance opening brace on l. 1"
        );
        assert_eq!(
            parse_template(r"parsing error \").unwrap_err(),
            "incomplete escape sequence on l. 1"
        );
    }
}
