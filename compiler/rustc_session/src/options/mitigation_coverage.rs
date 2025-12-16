use std::collections::BTreeSet;
use std::str::FromStr;

use rustc_macros::{BlobDecodable, Encodable};
use rustc_span::edition::Edition;
use rustc_target::spec::StackProtector;

use crate::Session;
use crate::config::Options;
use crate::options::CFGuard;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub enum DeniedPartialMitigationLevel {
    // Enabled(false) should be the bottom of the Ord hierarchy
    Enabled(bool),
    StackProtector(StackProtector),
}

impl DeniedPartialMitigationLevel {
    pub fn level_str(&self) -> &'static str {
        match self {
            DeniedPartialMitigationLevel::StackProtector(StackProtector::All) => "=all",
            DeniedPartialMitigationLevel::StackProtector(StackProtector::Basic) => "=basic",
            DeniedPartialMitigationLevel::StackProtector(StackProtector::Strong) => "=strong",
            // currently `=disabled` should not appear
            DeniedPartialMitigationLevel::Enabled(false) => "=disabled",
            DeniedPartialMitigationLevel::StackProtector(StackProtector::None)
            | DeniedPartialMitigationLevel::Enabled(true) => "",
        }
    }
}

impl std::fmt::Display for DeniedPartialMitigationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeniedPartialMitigationLevel::StackProtector(StackProtector::All) => {
                write!(f, "all")
            }
            DeniedPartialMitigationLevel::StackProtector(StackProtector::Basic) => {
                write!(f, "basic")
            }
            DeniedPartialMitigationLevel::StackProtector(StackProtector::Strong) => {
                write!(f, "strong")
            }
            DeniedPartialMitigationLevel::Enabled(true) => {
                write!(f, "enabled")
            }
            DeniedPartialMitigationLevel::StackProtector(StackProtector::None)
            | DeniedPartialMitigationLevel::Enabled(false) => {
                write!(f, "disabled")
            }
        }
    }
}

impl From<bool> for DeniedPartialMitigationLevel {
    fn from(value: bool) -> Self {
        DeniedPartialMitigationLevel::Enabled(value)
    }
}

impl From<StackProtector> for DeniedPartialMitigationLevel {
    fn from(value: StackProtector) -> Self {
        DeniedPartialMitigationLevel::StackProtector(value)
    }
}

pub struct DeniedPartialMitigationKindParseError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct MitigationCoverage {
    pub kind: DeniedPartialMitigationKind,
    pub enabled: bool,
}

macro_rules! intersperse {
    ($sep:expr, ($first:expr $(, $rest:expr)* $(,)?)) => {
        concat!($first $(, $sep, $rest)*)
    };
}

macro_rules! denied_partial_mitigations {
    ([$self:ident] enum $kind:ident {$(($name:ident, $text:expr, $since:ident, $code:expr)),*}) => {
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Encodable, BlobDecodable)]
        pub enum DeniedPartialMitigationKind {
            $($name),*
        }

        impl std::fmt::Display for DeniedPartialMitigationKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(DeniedPartialMitigationKind::$name => write!(f, $text)),*
                }
            }
        }

        impl DeniedPartialMitigationKind {
            pub(crate) const KINDS: &'static str = concat!("comma-separated list of mitigation kinds (available: ",
                intersperse!(", ", ($(concat!("`", $text, "`")),*)), ")");
        }

        impl FromStr for DeniedPartialMitigationKind {
            type Err = DeniedPartialMitigationKindParseError;

            fn from_str(v: &str) -> Result<DeniedPartialMitigationKind, DeniedPartialMitigationKindParseError> {
                match v {
                    $($text => Ok(DeniedPartialMitigationKind::$name)),*
                    ,
                    _ => Err(DeniedPartialMitigationKindParseError),
                }
            }
        }

        #[allow(unused)]
        impl DeniedPartialMitigationKind {
            pub fn enforced_since(&self) -> Edition {
                match self {
                    // Should change the enforced-since edition of StackProtector to 2015
                    // (all editions) when `-C stack-protector` is stabilized.
                    $(DeniedPartialMitigationKind::$name => Edition::$since),*
                }
            }
        }

        impl Options {
            pub fn all_denied_partial_mitigations(&self) -> impl Iterator<Item = DeniedPartialMitigationKind> {
                [$(DeniedPartialMitigationKind::$name),*].into_iter()
            }
        }

        impl Session {
            pub fn gather_enabled_denied_partial_mitigations(&$self) -> Vec<DeniedPartialMitigation> {
                let mut mitigations = [
                    $(
                    DeniedPartialMitigation {
                        kind: DeniedPartialMitigationKind::$name,
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

denied_partial_mitigations! {
    [self]
    enum DeniedPartialMitigationKind {
        (StackProtector, "stack-protector", EditionFuture, self.stack_protector()),
        (ControlFlowGuard, "control-flow-guard", EditionFuture, self.opts.cg.control_flow_guard == CFGuard::Checks)
    }
}

/// Denied-partial mitigations, see [RFC 3855](https://github.com/rust-lang/rfcs/pull/3855)
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct DeniedPartialMitigation {
    pub kind: DeniedPartialMitigationKind,
    pub level: DeniedPartialMitigationLevel,
}

impl Options {
    // Return the list of mitigations that are allowed to be partial
    pub fn allowed_partial_mitigations(
        &self,
        edition: Edition,
    ) -> impl Iterator<Item = DeniedPartialMitigationKind> {
        let mut result: BTreeSet<_> = self
            .all_denied_partial_mitigations()
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
