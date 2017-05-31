use check::path::Path;

use rustc::hir::def::Export;

use std::collections::BTreeSet;
use std::cmp::Ordering;

use syntax_pos::Span;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum ChangeCategory {
    Patch,
    NonBreaking,
    TechnicallyBreaking,
    Breaking,
}

pub use self::ChangeCategory::*;

pub enum ChangeType {
    Removal,
    Addition,
}

pub use self::ChangeType::*;

impl ChangeType {
    pub fn to_category(&self) -> ChangeCategory {
        match *self {
            Removal => Breaking,
            Addition => TechnicallyBreaking,
        }
    }
}

pub struct Change {
    change_type: ChangeType,
    path: Path,
    export: Export,
}

impl Change {
    pub fn span(&self) -> &Span {
        &self.export.span
    }

    pub fn type_(&self) -> &ChangeType {
        &self.change_type
    }
}

impl PartialEq for Change {
    fn eq(&self, other: &Change) -> bool {
        self.span() == other.span()
    }
}

impl Eq for Change {}

impl PartialOrd for Change {
    fn partial_cmp(&self, other: &Change) -> Option<Ordering> {
        self.span().partial_cmp(other.span())
    }
}

impl Ord for Change {
    fn cmp(&self, other: &Change) -> Ordering {
        self.span().cmp(other.span())
    }
}

pub struct ChangeSet {
    changes: BTreeSet<Change>,
    max: ChangeCategory,
}

impl ChangeSet {
    pub fn new() -> ChangeSet {
        ChangeSet {
            changes: BTreeSet::new(),
            max: Patch,
        }
    }

    pub fn add_change(&mut self, change: Change) {
        let cat = change.type_().to_category();

        if cat > self.max {
            self.max = cat;
        }

        self.changes.insert(change);
    }
}
