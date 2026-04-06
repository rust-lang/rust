use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Copy, Clone)]
struct MitigationStatus {
    // This is the index of the option in the command line. This is needed because
    // re-enabling a mitigation resets the partial mitigation status if it's later in the command
    // line, and this works across `-C` and `-Z` args.
    //
    // e.g. `-Z stack-protector=strong` resets `-C allow-partial-mitigations=stack-protector`.
    index: usize,
    allowed: Option<bool>,
}

#[derive(Clone, Default)]
pub struct MitigationCoverageMap {
    map: BTreeMap<DeniedPartialMitigationKind, MitigationStatus>,
}

impl MitigationCoverageMap {
    fn apply_mitigation(
        &mut self,
        kind: DeniedPartialMitigationKind,
        index: usize,
        allowed: Option<bool>,
    ) {
        self.map
            .entry(kind)
            .and_modify(|e| {
                if index >= e.index {
                    *e = MitigationStatus { index, allowed }
                }
            })
            .or_insert(MitigationStatus { index, allowed });
    }

    pub(crate) fn handle_allowdeny_mitigation_option(
        &mut self,
        v: Option<&str>,
        index: usize,
        allowed: bool,
    ) -> bool {
        match v {
            Some(s) => {
                for sub in s.split(',') {
                    match sub.parse() {
                        Ok(kind) => self.apply_mitigation(kind, index, Some(allowed)),
                        Err(_) => return false,
                    }
                }
                true
            }
            None => false,
        }
    }

    pub(crate) fn reset_mitigation(&mut self, kind: DeniedPartialMitigationKind, index: usize) {
        self.apply_mitigation(kind, index, None);
    }
}

pub struct DeniedPartialMitigationKindParseError;

macro_rules! intersperse {
    ($sep:expr, ($first:expr $(, $rest:expr)* $(,)?)) => {
        concat!($first $(, $sep, $rest)*)
    };
}

macro_rules! denied_partial_mitigations {
    ([$self:ident] enum $kind:ident {$(($name:ident, $text:expr, $since:ident, $code:expr)),*}) => {
        #[allow(non_camel_case_types)]
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
            pub fn allowed_by_default_at(&self, edition: Edition) -> bool {
                let denied_since = match self {
                    // Should change the denied-since edition of StackProtector to 2015
                    // (all editions) when `-C stack-protector` is stabilized.
                    $(DeniedPartialMitigationKind::$name => Edition::$since),*
                };
                edition < denied_since
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
        // The mitigation name should match the option name in rustc_session::options,
        // to allow for resetting the mitigation
        (stack_protector, "stack-protector", EditionFuture, self.stack_protector()),
        (control_flow_guard, "control-flow-guard", EditionFuture, self.opts.cg.control_flow_guard == CFGuard::Checks)
    }
}

/// A mitigation that cannot be partially enabled (see
/// [RFC 3855](https://github.com/rust-lang/rfcs/pull/3855)), but are currently enabled for this
/// crate.
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
            .filter(|mitigation| mitigation.allowed_by_default_at(edition))
            .collect();
        for (kind, MitigationStatus { index: _, allowed }) in &self.mitigation_coverage_map.map {
            match allowed {
                Some(true) => {
                    result.insert(*kind);
                }
                Some(false) => {
                    result.remove(kind);
                }
                None => {}
            }
        }
        result.into_iter()
    }
}
