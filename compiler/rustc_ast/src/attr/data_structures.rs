use std::fmt;

use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::{Span, Symbol};
use thin_vec::ThinVec;

use crate::attr::version::RustcVersion;

#[derive(Encodable, Decodable, Clone, Debug, PartialEq, Eq, Hash, HashStable_Generic)]
pub enum CfgEntry {
    All(ThinVec<CfgEntry>, Span),
    Any(ThinVec<CfgEntry>, Span),
    Not(Box<CfgEntry>, Span),
    Bool(bool, Span),
    NameValue { name: Symbol, value: Option<Symbol>, span: Span },
    Version(Option<RustcVersion>, Span),
}

impl CfgEntry {
    pub fn lower_spans(&mut self, lower_span: impl Copy + Fn(Span) -> Span) {
        match self {
            CfgEntry::All(subs, span) | CfgEntry::Any(subs, span) => {
                *span = lower_span(*span);
                subs.iter_mut().for_each(|sub| sub.lower_spans(lower_span));
            }
            CfgEntry::Not(sub, span) => {
                *span = lower_span(*span);
                sub.lower_spans(lower_span);
            }
            CfgEntry::Bool(_, span)
            | CfgEntry::NameValue { span, .. }
            | CfgEntry::Version(_, span) => {
                *span = lower_span(*span);
            }
        }
    }

    pub fn span(&self) -> Span {
        let (Self::All(_, span)
        | Self::Any(_, span)
        | Self::Not(_, span)
        | Self::Bool(_, span)
        | Self::NameValue { span, .. }
        | Self::Version(_, span)) = self;
        *span
    }

    /// Same as `PartialEq` but doesn't check spans and ignore order of cfgs.
    pub fn is_equivalent_to(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All(a, _), Self::All(b, _)) | (Self::Any(a, _), Self::Any(b, _)) => {
                a.len() == b.len() && a.iter().all(|a| b.iter().any(|b| a.is_equivalent_to(b)))
            }
            (Self::Not(a, _), Self::Not(b, _)) => a.is_equivalent_to(b),
            (Self::Bool(a, _), Self::Bool(b, _)) => a == b,
            (
                Self::NameValue { name: name1, value: value1, .. },
                Self::NameValue { name: name2, value: value2, .. },
            ) => name1 == name2 && value1 == value2,
            (Self::Version(a, _), Self::Version(b, _)) => a == b,
            _ => false,
        }
    }
}

impl fmt::Display for CfgEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn write_entries(
            name: &str,
            entries: &[CfgEntry],
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(f, "{name}(")?;
            for (nb, entry) in entries.iter().enumerate() {
                if nb != 0 {
                    f.write_str(", ")?;
                }
                entry.fmt(f)?;
            }
            f.write_str(")")
        }
        match self {
            Self::All(entries, _) => write_entries("all", entries, f),
            Self::Any(entries, _) => write_entries("any", entries, f),
            Self::Not(entry, _) => write!(f, "not({entry})"),
            Self::Bool(value, _) => write!(f, "{value}"),
            Self::NameValue { name, value, .. } => {
                match value {
                    // We use `as_str` and debug display to have characters escaped and `"`
                    // characters surrounding the string.
                    Some(value) => write!(f, "{name} = {:?}", value.as_str()),
                    None => write!(f, "{name}"),
                }
            }
            Self::Version(version, _) => match version {
                Some(version) => write!(f, "{version}"),
                None => Ok(()),
            },
        }
    }
}
