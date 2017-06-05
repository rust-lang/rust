use semcheck::path::ExportPath;

use rustc::hir::def::Export;
use rustc::session::Session;

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
/// regard the span and path of the associated item export. This allows us to sort them
/// by appearance in the source, but possibly could create conflict later on.
pub struct Change {
    /// The type of the change in question - see above.
    change_type: ChangeType,
    /// The export path this change was recorded for.
    path: ExportPath,
    /// The associated item's export.
    export: Export,
}

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
        self.span() == other.span() && self.path() == other.path()
    }
}

impl Eq for Change {}

impl PartialOrd for Change {
    fn partial_cmp(&self, other: &Change) -> Option<Ordering> {
        if let Some(ord1) = self.span().partial_cmp(other.span()) {
            if let Some(ord2) = self.path().partial_cmp(other.path()) {
                return Some(ord1.then(ord2));
            }
        }

        None
    }
}

impl Ord for Change {
    fn cmp(&self, other: &Change) -> Ordering {
        self.span().cmp(other.span()).then(self.path().cmp(other.path()))
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

impl ChangeSet {
    /// Add a change to the set and record it's category for later use.
    pub fn add_change(&mut self, change: Change) {
        let cat = change.type_().to_category();

        if cat > self.max {
            self.max = cat;
        }

        self.changes.insert(change);
    }

    /// Check for emptyness.
    ///
    /// Currently only used in tests.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Format the contents of a change set for user output.
    ///
    /// TODO: replace this with something more sophisticated.
    pub fn output(&self, session: &Session) {
        println!("max: {:?}", self.max);

        for change in &self.changes {
            println!("  {:?}: {}", change.type_(), change.path().inner());
            // span_note!(session, change.span(), "S0001");
            // session.span_warn(*change.span(), "change");
        }
    }
}

#[cfg(test)]
pub mod tests {
    use quickcheck::*;
    pub use super::*;

    use rustc::hir::def::Def;

    use std::cmp::{max, min};

    use syntax_pos::BytePos;
    use syntax_pos::hygiene::SyntaxContext;
    use syntax_pos::symbol::{Ident, Interner};

    #[derive(Clone, Debug)]
    pub struct Span_(Span);

    impl Span_ {
        pub fn inner(self) -> Span {
            self.0
        }
    }

    impl Arbitrary for Span_ {
        fn arbitrary<G: Gen>(g: &mut G) -> Span_ {
            let a: u32 = Arbitrary::arbitrary(g);
            let b: u32 = Arbitrary::arbitrary(g);
            Span_(Span {
                lo: BytePos(min(a, b)),
                hi: BytePos(max(a, b)),
                ctxt: SyntaxContext::empty()
            })
        }
    }

    impl Arbitrary for ChangeType {
        fn arbitrary<G: Gen>(g: &mut G) -> ChangeType {
            g.choose(&[Removal, Addition]).unwrap().clone()
        }
    }

    pub type Change_ = (ChangeType, Span_);

    /// We build these by hand, because symbols can't be sent between threads.
    fn build_change(t: ChangeType, s: Span) -> Change {
        let mut interner = Interner::new();
        let ident = Ident {
            name: interner.intern("test"),
            ctxt: SyntaxContext::empty(),
        };

        let export = Export {
            ident: ident,
            def: Def::Err,
            span: s,
        };

        Change::new(t, ExportPath::new(vec!["this is elegant enough".to_owned()]), export)
    }

    quickcheck! {
        /// The `Ord` instance of `Change` obeys transitivity.
        fn ord_change_transitive(c1: Change_, c2: Change_, c3: Change_) -> bool {
            let ch1 = build_change(c1.0, c1.1.inner());
            let ch2 = build_change(c2.0, c2.1.inner());
            let ch3 = build_change(c3.0, c3.1.inner());

            let mut res = true;

            if ch1 < ch2 && ch2 < ch3 {
                res &= ch1 < ch3;
            }

            if ch1 == ch2 && ch2 == ch3 {
                res &= ch1 == ch3;
            }

            if ch1 > ch2 && ch2 > ch3 {
                res &= ch1 > ch3;
            }

            res
        }

        /// The maximal change category for a change set gets computed correctly.
        fn max_change(changes: Vec<Change_>) -> bool {
            let mut set = ChangeSet::default();

            let max = changes.iter().map(|c| c.0.to_category()).max().unwrap_or(Patch);

            for &(ref change, ref span) in changes.iter() {
                set.add_change(build_change(change.clone(), span.clone().inner()));
            }

            set.max == max
        }

        /// Different spans imply different items.
        fn change_span_neq(c1: Change_, c2: Change_) -> bool {
            let s1 = c1.1.inner();
            let s2 = c2.1.inner();

            if s1 != s2 {
                let ch1 = build_change(c1.0, s1);
                let ch2 = build_change(c2.0, s2);

                ch1 != ch2
            } else {
                true
            }
        }

        /// Different paths imply different items.
        fn change_path_neq(c1: Change_, c2: ChangeType) -> bool {
            let span = c1.1.inner();
            let ch1 = build_change(c1.0, span.clone());
            let ch2 = build_change(c2, span);

            if ch1.path() != ch2.path() {
                ch1 != ch2
            } else {
                true
            }
        }
    }
}
