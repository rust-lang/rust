use semcheck::path::ExportPath;

use rustc::hir::def::Export;

use std::collections::BTreeSet;
use std::cmp::Ordering;

use syntax_pos::Span;

/// The categories we use when analyzing changes between crate versions.
///
/// These directly correspond to the semantic versioning spec, with the exception that
/// some breaking changes are categorized as "technically breaking" - that is, [1]
/// defines them as non-breaking when introduced to the standard libraries.
///
/// [1]: https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChangeCategory {
    /// Patch change - no change to the public API of a crate.
    Patch,
    /// A backwards-compatible change.
    NonBreaking,
    /// A breaking change that is very unlikely to cause breakage.
    TechnicallyBreaking,
    /// A breaking, backwards-incompatible change.
    Breaking,
}

pub use self::ChangeCategory::*;

impl Default for ChangeCategory {
    fn default() -> ChangeCategory {
        Patch
    }
}

/// The types of changes we identify.
///
/// TODO: This will be further extended in the future.
#[derive(Clone, Debug)]
pub enum ChangeType {
    /// The removal of a path an item is exported through.
    Removal,
    /// The addition of a path for an item (which possibly didn't exist previously).
    Addition,
}

pub use self::ChangeType::*;

impl ChangeType {
    /// Map a change type to the category it is part of.
    pub fn to_category(&self) -> ChangeCategory {
        match *self {
            Removal => Breaking,
            Addition => TechnicallyBreaking,
        }
    }
}

/// A change record.
///
/// Consists of all information we need to compute semantic versioning properties of
/// the change, as well as data we use to output it in a nice fashion.
///
/// It is important to note that the `Eq` and `Ord` instances are constucted to only
/// regard the span of the associated item export. This allows us to sort them by
/// appearance in the source, but possibly could create conflict later on.
/// TODO: decide about this.
pub struct Change {
    /// The type of the change in question - see above.
    change_type: ChangeType,
    /// The export path this change was recorded for.
    path: ExportPath,
    /// The associated item's export.
    export: Export,
}

// TODO: test the properties imposed by ord on all our custom impls

impl Change {
    pub fn new(change_type: ChangeType, path: ExportPath, export: Export) -> Change {
        Change {
            change_type: change_type,
            path: path,
            export: export,
        }
    }

    pub fn span(&self) -> &Span {
        &self.export.span
    }

    pub fn type_(&self) -> &ChangeType {
        &self.change_type
    }

    pub fn path(&self) -> &ExportPath {
        &self.path
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

/// The total set of changes recorded for two crate versions.
#[derive(Default)]
pub struct ChangeSet {
    /// The currently recorded changes.
    changes: BTreeSet<Change>,
    /// The most severe change category already recorded.
    max: ChangeCategory,
}

// TODO: test that the stored max is indeed the maximum of all categories

impl ChangeSet {
    /// Add a change to the set and record it's category for later use.
    pub fn add_change(&mut self, change: Change) {
        let cat = change.type_().to_category();

        if cat > self.max {
            self.max = cat;
        }

        self.changes.insert(change);
    }

    /// Format the contents of a change set for user output.
    ///
    /// TODO: replace this with something more sophisticated.
    pub fn output(&self) {
        println!("max: {:?}", self.max);

        for change in &self.changes {
            println!("  {:?}: {}", change.type_(), change.path().inner());
        }
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::*;
    use super::*;

    use rustc::hir::def::Def;

    use syntax_pos::DUMMY_SP;
    use syntax_pos::hygiene::SyntaxContext;
    use syntax_pos::symbol::{Ident, Interner};

    impl Arbitrary for ChangeType {
        fn arbitrary<G: Gen>(g: &mut G) -> ChangeType {
            g.choose(&[Removal, Addition]).unwrap().clone()
        }
    }

    /*
    impl Arbitrary for Change {
        fn arbitrary<G: Gen>(g: &mut G) -> Change {
            let mut interner = Interner::new();
            let export = Export {
                name: interner.intern("test"),
                def: Def::Err,
                span: DUMMY_SP,
            };
            Change::new(Arbitrary::arbitrary(g), Arbitrary::arbitrary(g), export)
        }
    }*/

    fn build_change(t: ChangeType) -> Change {
        let mut interner = Interner::new();
        let ident = Ident {
            name: interner.intern("test"),
            ctxt: SyntaxContext::empty(),
        };

        let export = Export {
            ident: ident,
            def: Def::Err,
            span: DUMMY_SP,
        };

        Change::new(t, ExportPath::new(vec!["this is elegant enough".to_owned()]), export)
    }

    quickcheck! {
        fn prop(changes: Vec<ChangeType>) -> bool {
            let mut set = ChangeSet::default();

            let max = changes.iter().map(|c| c.to_category()).max().unwrap_or(Patch);

            for change in changes.iter() {
                set.add_change(build_change(change.clone()));
            }

            set.max == max
        }
    }
}
