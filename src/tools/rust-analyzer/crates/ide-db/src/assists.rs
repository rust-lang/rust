//! This module defines the `Assist` data structure. The actual assist live in
//! the `ide_assists` downstream crate. We want to define the data structures in
//! this low-level crate though, because `ide_diagnostics` also need them
//! (fixits for diagnostics and assists are the same thing under the hood). We
//! want to compile `ide_assists` and `ide_diagnostics` in parallel though, so
//! we pull the common definitions upstream, to this crate.

use std::str::FromStr;

use syntax::TextRange;

use crate::{label::Label, source_change::SourceChange};

#[derive(Debug, Clone)]
pub struct Assist {
    pub id: AssistId,
    /// Short description of the assist, as shown in the UI.
    pub label: Label,
    pub group: Option<GroupLabel>,
    /// Target ranges are used to sort assists: the smaller the target range,
    /// the more specific assist is, and so it should be sorted first.
    pub target: TextRange,
    /// Computing source change sometimes is much more costly then computing the
    /// other fields. Additionally, the actual change is not required to show
    /// the lightbulb UI, it only is needed when the user tries to apply an
    /// assist. So, we compute it lazily: the API allow requesting assists with
    /// or without source change. We could (and in fact, used to) distinguish
    /// between resolved and unresolved assists at the type level, but this is
    /// cumbersome, especially if you want to embed an assist into another data
    /// structure, such as a diagnostic.
    pub source_change: Option<SourceChange>,
    pub trigger_signature_help: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistKind {
    // FIXME: does the None variant make sense? Probably not.
    None,

    QuickFix,
    Generate,
    Refactor,
    RefactorExtract,
    RefactorInline,
    RefactorRewrite,
}

impl AssistKind {
    pub fn contains(self, other: AssistKind) -> bool {
        if self == other {
            return true;
        }

        match self {
            AssistKind::None | AssistKind::Generate => true,
            AssistKind::Refactor => matches!(
                other,
                AssistKind::RefactorExtract
                    | AssistKind::RefactorInline
                    | AssistKind::RefactorRewrite
            ),
            _ => false,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            AssistKind::None => "None",
            AssistKind::QuickFix => "QuickFix",
            AssistKind::Generate => "Generate",
            AssistKind::Refactor => "Refactor",
            AssistKind::RefactorExtract => "RefactorExtract",
            AssistKind::RefactorInline => "RefactorInline",
            AssistKind::RefactorRewrite => "RefactorRewrite",
        }
    }
}

impl FromStr for AssistKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "None" => Ok(AssistKind::None),
            "QuickFix" => Ok(AssistKind::QuickFix),
            "Generate" => Ok(AssistKind::Generate),
            "Refactor" => Ok(AssistKind::Refactor),
            "RefactorExtract" => Ok(AssistKind::RefactorExtract),
            "RefactorInline" => Ok(AssistKind::RefactorInline),
            "RefactorRewrite" => Ok(AssistKind::RefactorRewrite),
            unknown => Err(format!("Unknown AssistKind: '{unknown}'")),
        }
    }
}

/// Unique identifier of the assist, should not be shown to the user
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistId(pub &'static str, pub AssistKind);

/// A way to control how many assist to resolve during the assist resolution.
/// When an assist is resolved, its edits are calculated that might be costly to always do by default.
#[derive(Debug)]
pub enum AssistResolveStrategy {
    /// No assists should be resolved.
    None,
    /// All assists should be resolved.
    All,
    /// Only a certain assist should be resolved.
    Single(SingleResolve),
}

/// Hold the [`AssistId`] data of a certain assist to resolve.
/// The original id object cannot be used due to a `'static` lifetime
/// and the requirement to construct this struct dynamically during the resolve handling.
#[derive(Debug)]
pub struct SingleResolve {
    /// The id of the assist.
    pub assist_id: String,
    // The kind of the assist.
    pub assist_kind: AssistKind,
}

impl AssistResolveStrategy {
    pub fn should_resolve(&self, id: &AssistId) -> bool {
        match self {
            AssistResolveStrategy::None => false,
            AssistResolveStrategy::All => true,
            AssistResolveStrategy::Single(single_resolve) => {
                single_resolve.assist_id == id.0 && single_resolve.assist_kind == id.1
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct GroupLabel(pub String);
