use std::borrow::Cow;

use jaq_core::ValT;

use crate::cache::Cache;

#[derive(Debug)]
pub enum Directive<'a> {
    /// `//@ eq <filter>`
    ///
    /// Executes `<filter>` on a documentation JSON file. `<filter>` must return 2 values. If
    /// they're equal to each other, a directive passes, i.e., just like with the `==` expression in
    /// `jq`.
    ///
    /// Note that any changes applied to a processed JSON object aren't saved between directives.
    /// To define cross-directive variables, use the `//@ arg` directive.
    ///
    /// An error occurs if `<filter>` produces less or more than 2 outputs.
    Eq,

    /// `//@ ne <filter>`
    ///
    /// Executes `<filter>` on a documentation JSON file. `<filter>` must return 2 values. If
    /// they're not equal to each other, a directive passes, i.e., just like with the `!=`
    /// expression in `jq`.
    ///
    /// Note that any changes applied to a processed JSON object aren't saved between directives.
    /// To define cross-directive variables, use the `//@ arg` directive.
    ///
    /// An error occurs if `<filter>` produces less or more than 2 outputs.
    Ne,

    /// `//@ jq <filter>`
    ///
    /// Executes `<filter>` on a documentation JSON file. `<filter>` must return a value other than
    /// `null` or [`false`] to mark that a directive has passed, i.e., just like with the
    /// "if-then-else-end" conditional in `jq`.
    ///
    /// Note that any changes applied to a processed JSON object aren't saved between directives.
    /// To define cross-directive variables, use the `//@ arg` directive.
    ///
    /// An error occurs if `<filter>` produces less or more than 1 output. For multiple outputs, use
    /// `jq`'s constructors to collect multiple outputs into one.
    Jq,

    /// `//@ arg <name> <filter>`
    ///
    /// Defines or redefines a global variable.
    ///
    /// Similar to `jq`'s `--arg` option but accepts `<filter>` instead and assigns a global
    /// variable to a `<filter>`'s output.
    ///
    /// An error occurs if `<filter>` produces 0, 2 or more outputs. For multiple outputs, use
    /// `jq`'s constructors to collect multiple outputs into one.
    Arg(&'a str),
}

impl<'a> Directive<'a> {
    /// Returns `Ok(None)` if the directive isn't from `jsondocck` (e.g., from `compiletest`).
    pub fn parse(directive: &str, args: &mut &'a str) -> Result<Option<Self>, String> {
        match directive {
            "arg" => {
                let Some((name, filter)) = args.trim_start().split_once(char::is_whitespace) else {
                    return Err("expected a name and a filter, received only the name".into());
                };

                *args = filter;

                Ok(Some(Self::Arg(name)))
            }
            "jq" => Ok(Some(Self::Jq)),
            "eq" => Ok(Some(Self::Eq)),
            "ne" => Ok(Some(Self::Ne)),
            // Ignore `compiletest` directives, like `//@ edition`.
            _ if KNOWN_DIRECTIVE_NAMES.contains(&directive) => Ok(None),
            _ => Err(format!("unknown directive `//@ {directive}`")),
        }
    }
}

impl Directive<'_> {
    /// Performs the actual work of processing a directive or ensuring it passes.
    pub fn process(self, cache: &mut Cache, args: &str) -> Result<(), Cow<'static, str>> {
        let filter = cache.filter(args)?;
        let mut values = filter.run(cache);
        let first = values.next()?;

        match self {
            Directive::Arg(name) => cache.arg(name, first),
            Directive::Jq => {
                if !first.as_bool() {
                    return Err("received `false` or `null`".into());
                }
            }
            Directive::Eq => {
                let right = values.next()?;

                if first != right {
                    return Err(format!(
                        "assertion `left == right` failed: left: {first} right: {right}"
                    )
                    .into());
                }
            }
            Directive::Ne => {
                let right = values.next()?;

                if first == right {
                    return Err(format!(
                        "assertion `left != right` failed: left: {first} right: {right}"
                    )
                    .into());
                }
            }
        }

        values.is_empty().map_err(Cow::from)
    }
}

// FIXME: This setup is temporary until we figure out how to improve this situation.
//        See <https://github.com/rust-lang/rust/issues/125813#issuecomment-2141953780>.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../compiletest/src/directive-list.rs"));
