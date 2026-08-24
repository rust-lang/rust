use itertools::Itertools as _;
use std::fmt::{self, Display};

pub struct ConfMetadata {
    pub name: &'static str,
    pub default: String,
    pub lints: &'static [&'static str],
    pub doc: &'static str,
    pub renamed_to: Option<&'static str>,
}

impl Display for ConfMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "- `{}`: {}", self.name, self.doc)?;
        if !self.default.is_empty() {
            write!(f, "\n\n   (default: `{}`)", self.default)?;
        }
        Ok(())
    }
}

impl ConfMetadata {
    pub fn display_markdown_paragraph(&self) -> impl '_ + Display {
        struct S<'a>(&'a ConfMetadata);
        impl Display for S<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "## `{}`\n{}\n\n**Default Value:** `{}`\n\n---\n**Affected lints:**\n{}\n\n",
                    self.0.name,
                    self.0
                        .doc
                        .lines()
                        .format_with("\n", |doc, f| f(&doc.strip_prefix(" ").unwrap_or(doc))),
                    self.0.default,
                    self.0.lints.iter().format_with("\n", |name, f| f(&format_args!(
                        "* [`{name}`](https://rust-lang.github.io/rust-clippy/master/index.html#{name})"
                    ))),
                )
            }
        }
        S(self)
    }

    pub fn display_markdown_link(&self) -> impl '_ + Display {
        struct S<'a>(&'a ConfMetadata);
        impl Display for S<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "[`{}`]: https://doc.rust-lang.org/clippy/lint_configuration.html#{}",
                    self.0.name, self.0.name,
                )
            }
        }
        S(self)
    }
}
