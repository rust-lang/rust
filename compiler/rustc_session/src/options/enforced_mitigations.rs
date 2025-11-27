use std::collections::BTreeSet;
use std::str::FromStr;

use rustc_macros::{BlobDecodable, Encodable};
use rustc_span::edition::Edition;
use rustc_target::spec::StackProtector;

use crate::Session;
use crate::config::Options;
use crate::options::CFGuard;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub enum EnforcedMitigationLevel {
    // Enabled(false) should be the bottom of the Ord hierarchy
    Enabled(bool),
    StackProtector(StackProtector),
}

impl EnforcedMitigationLevel {
    pub fn level_str(&self) -> &'static str {
        match self {
            EnforcedMitigationLevel::StackProtector(StackProtector::All) => "=all",
            EnforcedMitigationLevel::StackProtector(StackProtector::Basic) => "=basic",
            EnforcedMitigationLevel::StackProtector(StackProtector::Strong) => "=strong",
            // currently `=disabled` should not appear
            EnforcedMitigationLevel::Enabled(false) => "=disabled",
            EnforcedMitigationLevel::StackProtector(StackProtector::None)
            | EnforcedMitigationLevel::Enabled(true) => "",
        }
    }
}

impl std::fmt::Display for EnforcedMitigationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnforcedMitigationLevel::StackProtector(StackProtector::All) => {
                write!(f, "all")
            }
            EnforcedMitigationLevel::StackProtector(StackProtector::Basic) => {
                write!(f, "basic")
            }
            EnforcedMitigationLevel::StackProtector(StackProtector::Strong) => {
                write!(f, "strong")
            }
            EnforcedMitigationLevel::Enabled(true) => {
                write!(f, "enabled")
            }
            EnforcedMitigationLevel::StackProtector(StackProtector::None)
            | EnforcedMitigationLevel::Enabled(false) => {
                write!(f, "disabled")
            }
        }
    }
}

impl From<bool> for EnforcedMitigationLevel {
    fn from(value: bool) -> Self {
        EnforcedMitigationLevel::Enabled(value)
    }
}

impl From<StackProtector> for EnforcedMitigationLevel {
    fn from(value: StackProtector) -> Self {
        EnforcedMitigationLevel::StackProtector(value)
    }
}

pub struct EnforcedMitigationKindParseError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct MitigationEnablement {
    pub kind: EnforcedMitigationKind,
    pub enabled: bool,
}

macro_rules! intersperse {
    ($sep:expr, ($first:expr $(, $rest:expr)* $(,)?)) => {
        concat!($first $(, $sep, $rest)*)
    };
}

macro_rules! enforced_mitigations {
    ([$self:ident] enum $kind:ident {$(($name:ident, $text:expr, $since:ident, $code:expr)),*}) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encodable, BlobDecodable)]
        pub enum EnforcedMitigationKind {
            $($name),*
        }

        impl std::fmt::Display for EnforcedMitigationKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(EnforcedMitigationKind::$name => write!(f, $text)),*
                }
            }
        }

        impl EnforcedMitigationKind {
            pub(crate) const KINDS: &'static str = concat!("comma-separated list of mitigation kinds (available: ",
                intersperse!(", ", ($(concat!("`", $text, "`")),*)), ")");
        }

        impl FromStr for EnforcedMitigationKind {
            type Err = EnforcedMitigationKindParseError;

            fn from_str(v: &str) -> Result<EnforcedMitigationKind, EnforcedMitigationKindParseError> {
                match v {
                    $($text => Ok(EnforcedMitigationKind::$name)),*
                    ,
                    _ => Err(EnforcedMitigationKindParseError),
                }
            }
        }

        #[allow(unused)]
        impl EnforcedMitigationKind {
            pub fn enforced_since(&self) -> Edition {
                match self {
                    // Should change the enforced-since edition of StackProtector to 2015
                    // (all editions) when `-C stack-protector` is stabilized.
                    $(EnforcedMitigationKind::$name => Edition::$since),*
                }
            }
        }

        impl Options {
            pub fn all_enforced_mitigations(&self) -> impl Iterator<Item = EnforcedMitigationKind> {
                [$(EnforcedMitigationKind::$name),*].into_iter()
            }
        }

        impl Session {
            pub fn gather_enabled_enforced_mitigations(&$self) -> Vec<EnforcedMitigation> {
                let mut mitigations = [
                    $(
                    EnforcedMitigation {
                        kind: EnforcedMitigationKind::$name,
                        level: From::from($code),
                    }
                    ),*
                ];
                mitigations.sort();
                mitigations.into_iter().collect()
            }
        }
    }
}

enforced_mitigations! {
    [self]
    enum EnforcedMitigationKind {
        (StackProtector, "stack-protector", EditionFuture, self.stack_protector()),
        (ControlFlowGuard, "control-flow-guard", EditionFuture, self.opts.cg.control_flow_guard == CFGuard::Checks)
    }
}

/// Enforced mitigations, see [RFC 3855](https://github.com/rust-lang/rfcs/pull/3855)
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct EnforcedMitigation {
    pub kind: EnforcedMitigationKind,
    pub level: EnforcedMitigationLevel,
}

impl Options {
    // Return the list of mitigations that are allowed to be partial
    pub fn allowed_partial_mitigations(
        &self,
        edition: Edition,
    ) -> impl Iterator<Item = EnforcedMitigationKind> {
        let mut result: BTreeSet<_> = self
            .all_enforced_mitigations()
            .filter(|mitigation| mitigation.enforced_since() > edition)
            .collect();
        for mitigation in &self.unstable_opts.allow_partial_mitigations {
            if mitigation.enabled {
                result.insert(mitigation.kind);
            } else {
                result.remove(&mitigation.kind);
            }
        }
        result.into_iter()
    }
}
