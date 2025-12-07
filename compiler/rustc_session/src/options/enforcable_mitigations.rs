use std::collections::BTreeSet;
use std::str::FromStr;

use rustc_macros::{BlobDecodable, Encodable};
use rustc_span::edition::Edition;
use rustc_target::spec::StackProtector;

use crate::Session;
use crate::config::Options;
use crate::options::CFGuard;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub enum EnforcableMitigationLevel {
    // Enabled(false) should be the bottom of the Ord hierarchy
    Enabled(bool),
    StackProtector(StackProtector),
}

impl EnforcableMitigationLevel {
    pub fn level_str(&self) -> &'static str {
        match self {
            EnforcableMitigationLevel::StackProtector(StackProtector::All) => "=all",
            EnforcableMitigationLevel::StackProtector(StackProtector::Basic) => "=basic",
            EnforcableMitigationLevel::StackProtector(StackProtector::Strong) => "=strong",
            // currently `=disabled` should not appear
            EnforcableMitigationLevel::Enabled(false) => "=disabled",
            EnforcableMitigationLevel::StackProtector(StackProtector::None)
            | EnforcableMitigationLevel::Enabled(true) => "",
        }
    }
}

impl std::fmt::Display for EnforcableMitigationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnforcableMitigationLevel::StackProtector(StackProtector::All) => {
                write!(f, "all")
            }
            EnforcableMitigationLevel::StackProtector(StackProtector::Basic) => {
                write!(f, "basic")
            }
            EnforcableMitigationLevel::StackProtector(StackProtector::Strong) => {
                write!(f, "strong")
            }
            EnforcableMitigationLevel::Enabled(true) => {
                write!(f, "enabled")
            }
            EnforcableMitigationLevel::StackProtector(StackProtector::None)
            | EnforcableMitigationLevel::Enabled(false) => {
                write!(f, "disabled")
            }
        }
    }
}

impl From<bool> for EnforcableMitigationLevel {
    fn from(value: bool) -> Self {
        EnforcableMitigationLevel::Enabled(value)
    }
}

impl From<StackProtector> for EnforcableMitigationLevel {
    fn from(value: StackProtector) -> Self {
        EnforcableMitigationLevel::StackProtector(value)
    }
}

pub struct EnforcableMitigationKindParseError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct MitigationEnablement {
    pub kind: EnforcableMitigationKind,
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
        pub enum EnforcableMitigationKind {
            $($name),*
        }

        impl std::fmt::Display for EnforcableMitigationKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(EnforcableMitigationKind::$name => write!(f, $text)),*
                }
            }
        }

        impl EnforcableMitigationKind {
            pub(crate) const KINDS: &'static str = concat!("comma-separated list of mitigation kinds (available: ",
                intersperse!(", ", ($(concat!("`", $text, "`")),*)), ")");
        }

        impl FromStr for EnforcableMitigationKind {
            type Err = EnforcableMitigationKindParseError;

            fn from_str(v: &str) -> Result<EnforcableMitigationKind, EnforcableMitigationKindParseError> {
                match v {
                    $($text => Ok(EnforcableMitigationKind::$name)),*
                    ,
                    _ => Err(EnforcableMitigationKindParseError),
                }
            }
        }

        #[allow(unused)]
        impl EnforcableMitigationKind {
            pub fn enforced_since(&self) -> Edition {
                match self {
                    // Should change the enforced-since edition of StackProtector to 2015
                    // (all editions) when `-C stack-protector` is stabilized.
                    $(EnforcableMitigationKind::$name => Edition::$since),*
                }
            }
        }

        impl Options {
            pub fn all_enforced_mitigations(&self) -> impl Iterator<Item = EnforcableMitigationKind> {
                [$(EnforcableMitigationKind::$name),*].into_iter()
            }
        }

        impl Session {
            pub fn gather_enabled_enforcable_mitigations(&$self) -> Vec<EnforcableMitigation> {
                let mut mitigations = [
                    $(
                    EnforcableMitigation {
                        kind: EnforcableMitigationKind::$name,
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
    enum EnforcableMitigationKind {
        (StackProtector, "stack-protector", EditionFuture, self.stack_protector()),
        (ControlFlowGuard, "control-flow-guard", EditionFuture, self.opts.cg.control_flow_guard == CFGuard::Checks)
    }
}

/// Enforced mitigations, see [RFC 3855](https://github.com/rust-lang/rfcs/pull/3855)
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct EnforcableMitigation {
    pub kind: EnforcableMitigationKind,
    pub level: EnforcableMitigationLevel,
}

impl Options {
    // Return the list of mitigations that are allowed to be partial
    pub fn allowed_partial_mitigations(
        &self,
        edition: Edition,
    ) -> impl Iterator<Item = EnforcableMitigationKind> {
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
