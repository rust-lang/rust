use std::fmt;

#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::{Symbol, sym};

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
#[cfg_attr(feature = "nightly", derive(StableHash, Encodable, Decodable))]
pub struct LintName {
    /// The lint's tool name, if present.
    pub tool: Option<Symbol>,
    /// The lint's name, minus the tool name if any.
    ///
    /// For example:
    /// - `clippy::blah` -> `blah`
    /// - `unstable_imports` -> `unstable_imports`
    /// - `custom_tool::blah::bleh` -> `blah::bleh`
    pub name: Symbol,
}

impl LintName {
    pub fn new(segments: impl Iterator<Item = Symbol>) -> LintName {
        let segments = segments.collect::<Vec<_>>();
        let (tool, name) = match &*segments {
            [] => unreachable!(),
            [lint_name] => (None, *lint_name),
            [tool, lint_name] => (Some(*tool), *lint_name),
            // Not reachable for any builtin lints,
            // but reachable using `register_tool`.
            [tool, lint_name, rest @ ..] => {
                let mut lint_name = String::from(lint_name.as_str());
                for seg in rest {
                    lint_name.push_str("::");
                    lint_name.push_str(seg.as_str());
                }
                (Some(*tool), Symbol::intern(&lint_name))
            }
        };
        LintName { tool, name }
    }

    /// `lint` must not be empty.
    pub fn from_str(lint: &str) -> LintName {
        if lint.is_empty() {
            if cfg!(debug_assertions) {
                rustc_span::bug!("`lint` is empty")
            } else {
                return LintName { tool: None, name: sym::empty };
            }
        }
        if let Some((tool, name)) = lint.split_once("::") {
            LintName { tool: Some(Symbol::intern(tool)), name: Symbol::intern(name) }
        } else {
            LintName { tool: None, name: Symbol::intern(lint) }
        }
    }

    pub fn to_lowercase(mut self) -> LintName {
        fn lowercase(s: &mut Symbol) {
            let lower = s.as_str().to_lowercase();
            *s = Symbol::intern(&lower);
        }
        if let Some(tool) = &mut self.tool {
            lowercase(tool);
        }
        lowercase(&mut self.name);
        self
    }

    pub fn is(&self, lint: Symbol) -> bool {
        lint == self.name && self.tool.is_none()
    }
}

impl fmt::Display for LintName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tool) = self.tool {
            write!(f, "{tool}::")?
        }
        write!(f, "{}", self.name)?;
        Ok(())
    }
}

#[cfg(feature = "nightly")]
impl rustc_error_messages::IntoDiagArg for LintName {
    fn into_diag_arg(
        self,
        _: &mut Option<std::path::PathBuf>,
    ) -> rustc_error_messages::DiagArgValue {
        self.to_string().into_diag_arg(&mut None)
    }
}
