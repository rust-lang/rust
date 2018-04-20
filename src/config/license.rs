use std::fmt;
use std::fs::File;
use std::io;
use std::io::Read;

use regex;
use regex::Regex;

#[derive(Debug)]
pub enum LicenseError {
    IO(io::Error),
    Regex(regex::Error),
    Parse(String),
}

impl fmt::Display for LicenseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LicenseError::IO(ref err) => err.fmt(f),
            LicenseError::Regex(ref err) => err.fmt(f),
            LicenseError::Parse(ref err) => write!(f, "parsing failed, {}", err),
        }
    }
}

impl From<io::Error> for LicenseError {
    fn from(err: io::Error) -> LicenseError {
        LicenseError::IO(err)
    }
}

impl From<regex::Error> for LicenseError {
    fn from(err: regex::Error) -> LicenseError {
        LicenseError::Regex(err)
    }
}

// the template is parsed using a state machine
enum ParsingState {
    Lit,
    LitEsc,
    // the u32 keeps track of brace nesting
    Re(u32),
    ReEsc(u32),
    Abort(String),
}

use self::ParsingState::*;

pub struct TemplateParser {
    parsed: String,
    buffer: String,
    state: ParsingState,
    linum: u32,
    open_brace_line: u32,
}

impl TemplateParser {
    fn new() -> Self {
        Self {
            parsed: "^".to_owned(),
            buffer: String::new(),
            state: Lit,
            linum: 1,
            // keeps track of last line on which a regex placeholder was started
            open_brace_line: 0,
        }
    }

    /// Convert a license template into a string which can be turned into a regex.
    ///
    /// The license template could use regex syntax directly, but that would require a lot of manual
    /// escaping, which is inconvenient. It is therefore literal by default, with optional regex
    /// subparts delimited by `{` and `}`. Additionally:
    ///
    /// - to insert literal `{`, `}` or `\`, escape it with `\`
    /// - an empty regex placeholder (`{}`) is shorthand for `{.*?}`
    ///
    /// This function parses this input format and builds a properly escaped *string* representation
    /// of the equivalent regular expression. It **does not** however guarantee that the returned
    /// string is a syntactically valid regular expression.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(
    ///     TemplateParser::parse(
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
    pub fn parse(template: &str) -> Result<String, LicenseError> {
        let mut parser = Self::new();
        for chr in template.chars() {
            if chr == '\n' {
                parser.linum += 1;
            }
            parser.state = match parser.state {
                Lit => parser.trans_from_lit(chr),
                LitEsc => parser.trans_from_litesc(chr),
                Re(brace_nesting) => parser.trans_from_re(chr, brace_nesting),
                ReEsc(brace_nesting) => parser.trans_from_reesc(chr, brace_nesting),
                Abort(msg) => return Err(LicenseError::Parse(msg)),
            };
        }
        // check if we've ended parsing in a valid state
        match parser.state {
            Abort(msg) => return Err(LicenseError::Parse(msg)),
            Re(_) | ReEsc(_) => {
                return Err(LicenseError::Parse(format!(
                    "escape or balance opening brace on l. {}",
                    parser.open_brace_line
                )));
            }
            LitEsc => {
                return Err(LicenseError::Parse(format!(
                    "incomplete escape sequence on l. {}",
                    parser.linum
                )))
            }
            _ => (),
        }
        parser.parsed.push_str(&regex::escape(&parser.buffer));

        Ok(parser.parsed)
    }

    fn trans_from_lit(&mut self, chr: char) -> ParsingState {
        match chr {
            '{' => {
                self.parsed.push_str(&regex::escape(&self.buffer));
                self.buffer.clear();
                self.open_brace_line = self.linum;
                Re(1)
            }
            '}' => Abort(format!(
                "escape or balance closing brace on l. {}",
                self.linum
            )),
            '\\' => LitEsc,
            _ => {
                self.buffer.push(chr);
                Lit
            }
        }
    }

    fn trans_from_litesc(&mut self, chr: char) -> ParsingState {
        self.buffer.push(chr);
        Lit
    }

    fn trans_from_re(&mut self, chr: char, brace_nesting: u32) -> ParsingState {
        match chr {
            '{' => {
                self.buffer.push(chr);
                Re(brace_nesting + 1)
            }
            '}' => {
                match brace_nesting {
                    1 => {
                        // default regex for empty placeholder {}
                        if self.buffer.is_empty() {
                            self.parsed.push_str(".*?");
                        } else {
                            self.parsed.push_str(&self.buffer);
                        }
                        self.buffer.clear();
                        Lit
                    }
                    _ => {
                        self.buffer.push(chr);
                        Re(brace_nesting - 1)
                    }
                }
            }
            '\\' => {
                self.buffer.push(chr);
                ReEsc(brace_nesting)
            }
            _ => {
                self.buffer.push(chr);
                Re(brace_nesting)
            }
        }
    }

    fn trans_from_reesc(&mut self, chr: char, brace_nesting: u32) -> ParsingState {
        self.buffer.push(chr);
        Re(brace_nesting)
    }
}

pub fn load_and_compile_template(path: &str) -> Result<Regex, LicenseError> {
    let mut lt_file = File::open(&path)?;
    let mut lt_str = String::new();
    lt_file.read_to_string(&mut lt_str)?;
    let lt_parsed = TemplateParser::parse(&lt_str)?;
    Ok(Regex::new(&lt_parsed)?)
}

#[cfg(test)]
mod test {
    use super::TemplateParser;

    #[test]
    fn test_parse_license_template() {
        assert_eq!(
            TemplateParser::parse("literal (.*)").unwrap(),
            r"^literal \(\.\*\)"
        );
        assert_eq!(
            TemplateParser::parse(r"escaping \}").unwrap(),
            r"^escaping \}"
        );
        assert!(TemplateParser::parse("unbalanced } without escape").is_err());
        assert_eq!(
            TemplateParser::parse(r"{\d+} place{-?}holder{s?}").unwrap(),
            r"^\d+ place-?holders?"
        );
        assert_eq!(TemplateParser::parse("default {}").unwrap(), "^default .*?");
        assert_eq!(
            TemplateParser::parse(r"unbalanced nested braces {\{{3}}").unwrap(),
            r"^unbalanced nested braces \{{3}"
        );
        assert_eq!(
            &TemplateParser::parse("parsing error }")
                .unwrap_err()
                .to_string(),
            "parsing failed, escape or balance closing brace on l. 1"
        );
        assert_eq!(
            &TemplateParser::parse("parsing error {\nsecond line")
                .unwrap_err()
                .to_string(),
            "parsing failed, escape or balance opening brace on l. 1"
        );
        assert_eq!(
            &TemplateParser::parse(r"parsing error \")
                .unwrap_err()
                .to_string(),
            "parsing failed, incomplete escape sequence on l. 1"
        );
    }
}
