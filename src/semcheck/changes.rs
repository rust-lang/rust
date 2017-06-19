use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::session::Session;

use semver::Version;

use std::collections::{BTreeSet, BTreeMap, HashMap};
use std::cmp::Ordering;

use syntax::symbol::Ident;

use syntax_pos::Span;

/// The categories we use when analyzing changes between crate versions.
///
/// These directly correspond to the semantic versioning spec, with the exception that
/// some breaking changes are categorized as "technically breaking" - that is, [1]
/// defines them as non-breaking when introduced to the standard libraries.
///
/// [1]: https://github.com/rust-lang/rfcs/blob/master/text/1105-api-evolution.md
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

impl<'a> Default for ChangeCategory {
    fn default() -> ChangeCategory {
        Patch
    }
}

/// A change record of a change that introduced or removed an item.
///
/// It is important to note that the `Eq` and `Ord` instances are constucted to only
/// regard the span of the associated item export. This allows us to sort them by appearance
/// in the source, but possibly could create conflict later on.
// TODO: regard the origin of the span as well.
pub enum UnaryChange {
    /// An item has been added.
    Addition(Export),
    /// An item has been removed.
    Removal(Export),
}

impl UnaryChange {
    /// Get the change's category.
    pub fn to_category(&self) -> ChangeCategory {
        match *self {
            UnaryChange::Addition(_) => TechnicallyBreaking,
            UnaryChange::Removal(_) => Breaking,
        }
    }

    /// Get the change item's sole export.
    fn export(&self) -> &Export {
        match *self {
            UnaryChange::Addition(ref e) | UnaryChange::Removal(ref e) => e,
        }
    }

    /// Get the change item's sole span.
    pub fn span(&self) -> &Span {
        &self.export().span
    }

    /// Get the change item's ident.
    pub fn ident(&self) -> &Ident {
        &self.export().ident
    }

    /// Render the change's type to a string.
    pub fn type_(&self) -> &'static str {
        match *self {
            UnaryChange::Addition(_) => "Addition",
            UnaryChange::Removal(_) => "Removal",
        }
    }

    /// Report the change.
    fn report(&self, session: &Session) {
        let msg = format!("{} of at least one path to `{}` ({:?})",
                          self.type_(),
                          self.ident(),
                          self.to_category());
        let mut builder = session.struct_span_warn(self.export().span, &msg);

        builder.emit();
    }
}

impl PartialEq for UnaryChange {
    fn eq(&self, other: &UnaryChange) -> bool {
        self.span() == other.span()
    }
}

impl Eq for UnaryChange {}

impl PartialOrd for UnaryChange {
    fn partial_cmp(&self, other: &UnaryChange) -> Option<Ordering> {
        self.span().partial_cmp(other.span())
    }
}

impl Ord for UnaryChange {
    fn cmp(&self, other: &UnaryChange) -> Ordering {
        self.span().cmp(other.span())
    }
}

/// The types of changes we identify between items present in both crate versions.
// TODO: this needs a lot of refinement still
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinaryChangeType {
    /// An item has changed it's kind.
    KindDifference,
    /// A region parameter has been added to an item.
    RegionParameterAdded,
    /// A region parameter has been removed from an item.
    RegionParameterRemoved,
    /// A type parameter has been added to an item.
    TypeParameterAdded { defaulted: bool },
    /// A type parameter has been removed from an item.
    TypeParameterRemoved { defaulted: bool },
    /// The bounds on a type parameter have been loosened.
    TypeGeneralization,
    /// The bounds on a type parameter have been tightened.
    TypeSpecialization,
    /// A field has been added to a struct.
    StructFieldAdded { public: bool, total_public: bool }, // TODO: EXXXXPPPPLAAAAIN!
    /// A field has been removed from a struct.
    StructFieldRemoved { public: bool, total_public: bool }, // TODO: EXXXXPPPPLAAAIN!
    /// A struct has changed it's style.
    StructStyleChanged { now_tuple: bool, total_private: bool },
    /// A variant has been added to an enum.
    EnumVariantAdded,
    /// A variant has been removed from an enum.
    EnumVariantRemoved,
    /// A field hasb been added to an enum variant.
    VariantFieldAdded,
    /// A field has been removed from an enum variant.
    VariantFieldRemoved,
    /// An enum variant has changed it's style.
    VariantFieldStyleChanged { now_tuple: bool },
    /// A field in a struct or enum has changed it's type.
    FieldTypeChanged(String), // FIXME: terrible for obvious reasons
    /// An impl item has been added.
    TraitImplItemAdded { defaulted: bool }, // TODO: EXPLAAAIN!
    /// An impl item has been removed.
    TraitImplItemRemoved,
    /// An unknown change is any change we don't yet explicitly handle.
    Unknown,
}

pub use self::BinaryChangeType::*;

impl BinaryChangeType {
    // TODO: this will need a lot of changes (it's *very* conservative rn)
    pub fn to_category(&self) -> ChangeCategory {
        match *self {
            KindDifference |
            RegionParameterAdded |
            RegionParameterRemoved |
            TypeParameterAdded { defaulted: false } |
            TypeParameterRemoved { .. } |
            TypeSpecialization |
            StructFieldAdded { .. } |
            StructFieldRemoved { .. } |
            StructStyleChanged { .. } |
            EnumVariantAdded |
            EnumVariantRemoved |
            VariantFieldAdded |
            VariantFieldRemoved |
            VariantFieldStyleChanged { .. } |
            FieldTypeChanged(_) |
            TraitImplItemAdded { .. } |
            TraitImplItemRemoved |
            Unknown => Breaking,
            TypeGeneralization => TechnicallyBreaking,
            TypeParameterAdded { defaulted: true } => NonBreaking,
        }
    }
}

/// A change record of an item kept between versions.
///
/// It is important to note that the `Eq` and `Ord` instances are constucted to only
/// regard the *new* span of the associated item export. This allows us to sort them
/// by appearance in the *new* source, but possibly could create conflict later on.
// TODO: we should introduce an invariant that the two exports present are *always*
// tied together.
pub struct BinaryChange {
    /// The type of the change affecting the item.
    changes: BTreeSet<BinaryChangeType>,
    /// The most severe change category already recorded for the item.
    max: ChangeCategory,
    /// The old export of the change item.
    old: Export,
    /// The new export of the change item.
    new: Export,
}

impl BinaryChange {
    /// Construct a new empty change record for an item.
    fn new(old: Export, new: Export) -> BinaryChange {
        BinaryChange {
            changes: BTreeSet::new(),
            max: ChangeCategory::default(),
            old: old,
            new: new,
        }
    }

    /// Add another change type to the change record.
    fn add(&mut self, type_: BinaryChangeType) {
        let cat = type_.to_category();

        if cat > self.max {
            self.max = cat;
        }

        self.changes.insert(type_);
    }

    /// Get the change's category.
    fn to_category(&self) -> ChangeCategory {
        self.max.clone()
    }

    /// Get the new span of the change item.
    fn new_span(&self) -> &Span {
        &self.new.span
    }

    /// Get the ident of the change item.
    fn ident(&self) -> &Ident {
        &self.old.ident
    }

    /// Report the change.
    fn report(&self, session: &Session) {
        let msg = format!("{:?} changes in `{}`", self.max, self.ident());
        let mut builder = session.struct_span_warn(self.new.span, &msg);

        for change in &self.changes {
            let cat = change.to_category();
            let sub_msg = format!("{:?} ({:?})", cat, change);
            if cat == Breaking {
                builder.warn(&sub_msg);
            } else {
                builder.note(&sub_msg);
            }
        }

        builder.emit();
    }
}

impl PartialEq for BinaryChange {
    fn eq(&self, other: &BinaryChange) -> bool {
        self.new_span() == other.new_span()
    }
}

impl Eq for BinaryChange {}

impl PartialOrd for BinaryChange {
    fn partial_cmp(&self, other: &BinaryChange) -> Option<Ordering> {
        self.new_span().partial_cmp(other.new_span())
    }
}

impl Ord for BinaryChange {
    fn cmp(&self, other: &BinaryChange) -> Ordering {
        self.new_span().cmp(other.new_span())
    }
}

/// A change record for any item.
///
/// Consists of all information we need to compute semantic versioning properties of
/// the change(s) performed on it, as well as data we use to output it in a nice fashion.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Change {
    /// A wrapper around a unary change.
    Unary(UnaryChange),
    /// A wrapper around a binary change set.
    Binary(BinaryChange),
}

impl Change {
    /// Construct a new addition-change for the given export.
    fn new_addition(item: Export) -> Change {
        Change::Unary(UnaryChange::Addition(item))
    }

    /// Construct a new removal-change for the given export.
    fn new_removal(item: Export) -> Change {
        Change::Unary(UnaryChange::Removal(item))
    }

    /// Construct a new binary change for the given exports.
    fn new_binary(old: Export, new: Export) -> Change {
        Change::Binary(BinaryChange::new(old, new))
    }

    /// Add a change type to a given binary change.
    fn add(&mut self, type_: BinaryChangeType) {
        match *self {
            Change::Unary(_) => panic!("can't add binary change types to unary change"),
            Change::Binary(ref mut b) => b.add(type_),
        }
    }

    /// Get the change's representative span.
    fn span(&self) -> &Span {
        match *self {
            Change::Unary(ref u) => u.span(),
            Change::Binary(ref b) => b.new_span(),
        }
    }

    /// Get the change's category.
    fn to_category(&self) -> ChangeCategory {
        match *self {
            Change::Unary(ref u) => u.to_category(),
            Change::Binary(ref b) => b.to_category(),
        }
    }

    /// Report the change.
    fn report(&self, session: &Session) {
        match *self {
            Change::Unary(ref u) => u.report(session),
            Change::Binary(ref b) => b.report(session),
        }
    }
}

/// An identifier used to unambiguously refer to items we record changes for.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ChangeKey {
    /// An item referred to using the old definition's id.
    /// This includes items that have been removed *or* changed.
    OldKey(DefId),
    /// An item referred to using the new definition's id.
    /// This includes items that have been added *only*
    NewKey(DefId),
}

/// The total set of changes recorded for two crate versions.
#[derive(Default)]
pub struct ChangeSet {
    /// The currently recorded changes.
    changes: HashMap<ChangeKey, Change>,
    /// The mapping of spans to changes, for ordering purposes.
    spans: BTreeMap<Span, ChangeKey>,
    /// The most severe change category already recorded.
    max: ChangeCategory,
}

impl ChangeSet {
    /// Add a new addition-change for the given export.
    pub fn new_addition(&mut self, item: Export) {
        self.new_unary_change(Change::new_addition(item), ChangeKey::NewKey(item.def.def_id()));
    }

    /// Add a new removal-change for the given export.
    pub fn new_removal(&mut self, item: Export) {
        self.new_unary_change(Change::new_removal(item), ChangeKey::NewKey(item.def.def_id()));
    }

    /// Add a new (unary) change for the given key.
    fn new_unary_change(&mut self, change: Change, key: ChangeKey) {
        let cat = change.to_category();

        if cat > self.max {
            self.max = cat.clone();
        }

        self.spans.insert(*change.span(), key.clone());
        self.changes.insert(key, change);
    }

    /// Add a new binary change for the given exports.
    pub fn new_binary(&mut self, type_: BinaryChangeType, old: Export, new: Export) {
        let key = ChangeKey::OldKey(old.def.def_id());
        let cat = type_.to_category();

        if cat > self.max {
            self.max = cat.clone();
        }

        let entry = self.changes
            .entry(key)
            .or_insert_with(|| Change::new_binary(old, new));

        entry.add(type_);
    }

    pub fn add_binary(&mut self, type_: BinaryChangeType, old: DefId) -> bool {
        use std::collections::hash_map::Entry;

        let key = ChangeKey::OldKey(old);
        let cat = type_.to_category();

        if cat > self.max {
            self.max = cat.clone();
        }

        match self.changes.entry(key) {
            Entry::Occupied(c) => {
                c.into_mut().add(type_);
                true
            },
            Entry::Vacant(_) => false,
        }
    }

    /// Check whether an item with the given id has undergone breaking changes.
    ///
    /// The expected `DefId` is obviously an *old* one.
    pub fn item_breaking(&self, key: DefId) -> bool {
        // we only care about items that were present before, since only those can get breaking
        // changes (additions don't count).
        self.changes
            .get(&ChangeKey::OldKey(key))
            .map(|changes| changes.to_category() == Breaking)
            .unwrap_or(false)
    }

    /// Format the contents of a change set for user output.
    pub fn output(&self, session: &Session, version: &str) {
        if let Ok(mut new_version) = Version::parse(version) {
            match self.max {
                Patch => new_version.increment_patch(),
                NonBreaking | TechnicallyBreaking => new_version.increment_minor(),
                Breaking => new_version.increment_major(),
            }

            println!("version bump: {} -> ({:?}) -> {}", version, self.max, new_version);
        } else {
            println!("max change: {} (could not parse) -> {:?}", version, self.max);
        }

        for key in self.spans.values() {
            self.changes[key].report(session);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use quickcheck::*;
    pub use super::*;

    use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
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
                      ctxt: SyntaxContext::empty(),
                  })
        }
    }

    // we don't need this type elsewhere, so we define it here.
    #[derive(Clone, Debug)]
    pub enum UnaryChangeType {
        Removal,
        Addition,
    }

    impl Arbitrary for UnaryChangeType {
        fn arbitrary<G: Gen>(g: &mut G) -> UnaryChangeType {
            g.choose(&[UnaryChangeType::Removal, UnaryChangeType::Addition]).unwrap().clone()
        }
    }

    impl<'a> From<&'a UnaryChangeType> for ChangeCategory {
        fn from(change: &UnaryChangeType) -> ChangeCategory {
            match *change {
                UnaryChangeType::Addition => TechnicallyBreaking,
                UnaryChangeType::Removal => Breaking,
            }
        }
    }

    impl Arbitrary for BinaryChangeType {
        fn arbitrary<G: Gen>(g: &mut G) -> BinaryChangeType {
            g.choose(&[BinaryChangeType::Unknown]).unwrap().clone()
        }
    }

    pub type UnaryChange_ = (UnaryChangeType, Span_);

    /// We build these by hand, because symbols can't be sent between threads.
    fn build_unary_change(t: UnaryChangeType, s: Span) -> UnaryChange {
        let mut interner = Interner::new();
        let ident = Ident {
            name: interner.intern("test"),
            ctxt: SyntaxContext::empty(),
        };
        let export = Export {
            ident: ident,
            def: Def::Mod(DefId {
                krate: LOCAL_CRATE,
                index: CRATE_DEF_INDEX,
            }),
            span: s,
        };

        match t {
            UnaryChangeType::Addition => UnaryChange::Addition(export),
            UnaryChangeType::Removal => UnaryChange::Removal(export),
        }
    }

    pub type BinaryChange_ = (BinaryChangeType, Span_, Span_);

    fn build_binary_change(t: BinaryChangeType, s1: Span, s2: Span) -> BinaryChange {
        let mut interner = Interner::new();
        let ident1 = Ident {
            name: interner.intern("test"),
            ctxt: SyntaxContext::empty(),
        };
        let ident2 = Ident {
            name: interner.intern("test"),
            ctxt: SyntaxContext::empty(),
        };
        let export1 = Export {
            ident: ident1,
            def: Def::Mod(DefId {
                krate: LOCAL_CRATE,
                index: CRATE_DEF_INDEX,
            }),
            span: s1,
        };
        let export2 = Export {
            ident: ident2,
            def: Def::Mod(DefId {
                krate: LOCAL_CRATE,
                index: CRATE_DEF_INDEX,
            }),
            span: s2,
        };

        let mut change = BinaryChange::new(export1, export2);
        change.add(t);

        change
    }

    quickcheck! {
        /// The `Ord` instance of `Change` is transitive.
        fn ord_uchange_transitive(c1: UnaryChange_, c2: UnaryChange_, c3: UnaryChange_) -> bool {
            let ch1 = build_unary_change(c1.0, c1.1.inner());
            let ch2 = build_unary_change(c2.0, c2.1.inner());
            let ch3 = build_unary_change(c3.0, c3.1.inner());

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

        fn ord_bchange_transitive(c1: BinaryChange_, c2: BinaryChange_, c3: BinaryChange_)
            -> bool
        {
            let ch1 = build_binary_change(c1.0, c1.1.inner(), c1.2.inner());
            let ch2 = build_binary_change(c2.0, c2.1.inner(), c2.2.inner());
            let ch3 = build_binary_change(c3.0, c3.1.inner(), c3.2.inner());

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
        fn max_uchange(changes: Vec<UnaryChange_>) -> bool {
            let mut set = ChangeSet::default();

            let max = changes.iter().map(|c| From::from(&c.0)).max().unwrap_or(Patch);

            for &(ref change, ref span) in changes.iter() {
                let change = build_unary_change(change.clone(), span.clone().inner());
                let key = ChangeKey::NewKey(change.export().def.def_id());
                set.new_unary_change(Change::Unary(change), key);
            }

            set.max == max
        }

        /// The maximal change category for a change set gets computed correctly.
        fn max_bchange(changes: Vec<BinaryChange_>) -> bool {
            let mut set = ChangeSet::default();

            let max = changes.iter().map(|c| c.0.to_category()).max().unwrap_or(Patch);

            for &(ref change, ref span1, ref span2) in changes.iter() {
                let change =
                    build_binary_change(change.clone(),
                                        span1.clone().inner(),
                                        span2.clone().inner());
                let key = ChangeKey::OldKey(change.old.def.def_id());
                set.new_unary_change(Change::Binary(change), key);
            }

            set.max == max
        }

        /// Difference in spans implies difference in `Change`s.
        fn uchange_span_neq(c1: UnaryChange_, c2: UnaryChange_) -> bool {
            let s1 = c1.1.inner();
            let s2 = c2.1.inner();

            if s1 != s2 {
                let ch1 = build_unary_change(c1.0, s1);
                let ch2 = build_unary_change(c2.0, s2);

                ch1 != ch2
            } else {
                true
            }
        }

        /// Difference in spans implies difference in `Change`s.
        fn bchange_span_neq(c1: BinaryChange_, c2: BinaryChange_) -> bool {
            let s1 = c1.1.inner();
            let s2 = c2.1.inner();

            if s1 != s2 {
                let ch1 = build_binary_change(c1.0, c1.2.inner(), s1);
                let ch2 = build_binary_change(c2.0, c2.2.inner(), s2);

                ch1 != ch2
            } else {
                true
            }
        }
    }
}
