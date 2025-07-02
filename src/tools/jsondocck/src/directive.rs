use std::borrow::Cow;

use jaq_core::ValT;

use crate::cache::Cache;

#[derive(Debug)]
pub enum Directive<'a> {
    /// `//@ jq <filter>`
    ///
    /// Executes `<filter>` on a documentation JSON file. `<filter>` must return a value other than
    /// `null` or [`false`] to mark that a directive has passed, i.e., just like with the
    /// "if-then-else-end" conditional in `jq`.
    ///
    /// Note that any changes applied to a processed JSON object aren't saved between directives.
    /// To define cross-directive variables, use the `//@ arg` directive.
    Jq(&'a str),

    /// `//@ arg <name> <filter>`
    ///
    /// Defines or redefines a global variable.
    ///
    /// Similar to `jq`'s `--arg` option but accepts `<filter>` instead and assigns a global
    /// variable to a `<filter>`'s output.
    ///
    /// An error occurs if `<filter>` produces 0, 2 or more outputs. For multiple outputs, use
    /// `jq`'s constructors to collect multiple outputs into one.
    Arg { name: &'a str, filter: &'a str },
}

impl<'a> Directive<'a> {
    /// Returns `Ok(None)` if the directive isn't from `jsondocck` (e.g., from `compiletest`).
    pub fn parse(directive: &str, args: &'a str) -> Result<Option<Self>, String> {
        match directive {
            "arg" => {
                let Some((name, filter)) = args.trim_start().split_once(char::is_whitespace) else {
                    return Err("expected a name and a filter, received only the name".into());
                };

                Ok(Some(Self::Arg { name, filter }))
            }
            "jq" => Ok(Some(Self::Jq(args))),
            // Ignore `compiletest` directives, like `//@ edition`.
            _ if KNOWN_DIRECTIVE_NAMES.contains(&directive) => Ok(None),
            _ => Err(format!("unknown directive `//@ {directive}`")),
        }
    }
}

impl Directive<'_> {
    /// Performs the actual work of processing a directive or ensuring it passes.
    pub fn process(self, cache: &mut Cache) -> Result<(), Cow<'static, str>> {
        match self {
            Directive::Arg { name, filter } => cache.arg(name, cache.filter(filter)?),
            Directive::Jq(filter) => {
                if !cache.filter(filter)?.as_bool() {
                    return Err("received `false` or `null`".into());
                }
            }
        }

        Ok(())
    }
}

// FIXME: This setup is temporary until we figure out how to improve this situation.
//        See <https://github.com/rust-lang/rust/issues/125813#issuecomment-2141953780>.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../compiletest/src/directive-list.rs"));
