//! Change representation.
//!
//! This module provides data types to represent, store and record changes found in various
//! analysis passes. We distinguish between "unary" and "binary" changes, depending on whether
//! there is a single item affected in one crate version or a matching item in the other crate
//! version as well. The ordering of changes and output generation is performed using these data
//! structures, too.

use rustc::hir::def::Export;
use rustc::hir::def_id::DefId;
use rustc::session::Session;
use rustc::ty::error::TypeError;

use semver::Version;

use std::collections::{BTreeMap, HashMap};
use std::cmp::Ordering;

use syntax::symbol::{Ident, Symbol};

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

/// A change record of a newly introduced or removed item.
///
/// It is important to note that the `Eq` and `Ord` instances are constucted to only
/// regard the span of the associated item export.
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
#[derive(Clone, Debug)]
pub enum BinaryChangeType<'tcx> {
    /// An item has been made public.
    ItemMadePublic,
    /// An item has been made private
    ItemMadePrivate,
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
    /// A variant has been added to an enum.
    VariantAdded,
    /// A variant has been removed from an enum.
    VariantRemoved,
    /// A field hasb been added to a variant.
    VariantFieldAdded { public: bool, total_public: bool },
    /// A field has been removed from a variant.
    VariantFieldRemoved { public: bool, total_public: bool },
    /// A variant has changed it's style.
    VariantStyleChanged { now_struct: bool, total_private: bool },
    /// A function has changed it's constness.
    FnConstChanged { now_const: bool },
    /// A method has changed whether it can be invoked as a method call.
    MethodSelfChanged { now_self: bool },
    /// A trait's definition added an item.
    TraitItemAdded { defaulted: bool },
    /// A trait's definition removed an item.
    TraitItemRemoved { defaulted: bool },
    /// A trait's definition changed it's unsafety.
    TraitUnsafetyChanged { now_unsafe: bool },
    /// A field in a struct or enum has changed it's type.
    TypeChanged { error: TypeError<'tcx> },
    /// An unknown change is any change we don't yet explicitly handle.
    Unknown,
}

pub use self::BinaryChangeType::*;

impl<'tcx> BinaryChangeType<'tcx> {
    pub fn to_category(&self) -> ChangeCategory {
        match *self {
            ItemMadePrivate |
            KindDifference |
            RegionParameterAdded |
            RegionParameterRemoved |
            TypeParameterAdded { defaulted: false } |
            TypeParameterRemoved { .. } |
            VariantAdded |
            VariantRemoved |
            VariantFieldAdded { .. } |
            VariantFieldRemoved { .. } |
            VariantStyleChanged { .. } |
            TypeChanged { .. } |
            FnConstChanged { now_const: false } |
            MethodSelfChanged { now_self: false } |
            TraitItemAdded { defaulted: false } |
            TraitItemRemoved { .. } |
            TraitUnsafetyChanged { .. } |
            Unknown => Breaking,
            MethodSelfChanged { now_self: true } |
            TraitItemAdded { defaulted: true } |
            ItemMadePublic => TechnicallyBreaking,
            TypeParameterAdded { defaulted: true } |
            FnConstChanged { now_const: true } => NonBreaking,
        }
    }
}

/// A change record of an item kept between versions.
///
/// It is important to note that the `Eq` and `Ord` instances are constucted to only
/// regard the *new* span of the associated item export. This allows us to sort them
/// by appearance in the *new* source.
pub struct BinaryChange<'tcx> {
    /// The type of the change affecting the item.
    changes: Vec<(BinaryChangeType<'tcx>, Option<Span>)>,
    /// The most severe change category already recorded for the item.
    max: ChangeCategory,
    /// The symbol associated with the change item.
    name: Symbol,
    /// The new span associated with the change item.
    new_span: Span,
    /// Whether to output changes. Used to distinguish all-private items.
    output: bool
}

impl<'tcx> BinaryChange<'tcx> {
    /// Construct a new empty change record for an item.
    fn new(name: Symbol, span: Span, output: bool) -> BinaryChange<'tcx> {
        BinaryChange {
            changes: Vec::new(),
            max: ChangeCategory::default(),
            name: name,
            new_span: span,
            output: output,
        }
    }

    /// Add another change type to the change record.
    fn add(&mut self, type_: BinaryChangeType<'tcx>, span: Option<Span>) {
        let cat = type_.to_category();

        if cat > self.max {
            self.max = cat;
        }

        self.changes.push((type_, span));
    }

    /// Get the change's category.
    fn to_category(&self) -> ChangeCategory {
        self.max.clone()
    }

    /// Get the new span of the change item.
    fn new_span(&self) -> &Span {
        &self.new_span
    }

    /// Get the ident of the change item.
    fn ident(&self) -> &Symbol {
        &self.name
    }

    /// Report the change.
    fn report(&self, session: &Session) {
        if self.max == Patch || !self.output {
            return;
        }

        let msg = format!("{:?} changes in `{}`", self.max, self.ident());
        let mut builder = session.struct_span_warn(self.new_span, &msg);

        for change in &self.changes {
            let cat = change.0.to_category();
            let sub_msg = format!("{:?} ({:?})", cat, change.0);
            if let Some(span) = change.1 {
                if cat == Breaking {
                    builder.span_warn(span, &sub_msg);
                } else {
                    builder.span_note(span, &sub_msg,);
                }
            // change.1 == None from here on.
            } else if cat == Breaking {
                builder.warn(&sub_msg);
            } else {
                builder.note(&sub_msg);
            }
        }

        builder.emit();
    }
}

impl<'tcx> PartialEq for BinaryChange<'tcx> {
    fn eq(&self, other: &BinaryChange) -> bool {
        self.new_span() == other.new_span()
    }
}

impl<'tcx> Eq for BinaryChange<'tcx> {}

impl<'tcx> PartialOrd for BinaryChange<'tcx> {
    fn partial_cmp(&self, other: &BinaryChange<'tcx>) -> Option<Ordering> {
        self.new_span().partial_cmp(other.new_span())
    }
}

impl<'tcx> Ord for BinaryChange<'tcx> {
    fn cmp(&self, other: &BinaryChange<'tcx>) -> Ordering {
        self.new_span().cmp(other.new_span())
    }
}

/// A change record for any item.
///
/// Consists of all information we need to compute semantic versioning properties of
/// the change(s) performed on it, as well as data we use to output it in a nice fashion.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Change<'tcx> {
    /// A wrapper around a unary change.
    Unary(UnaryChange),
    /// A wrapper around a binary change set.
    Binary(BinaryChange<'tcx>),
}

impl<'tcx> Change<'tcx> {
    /// Construct a new addition-change for the given export.
    fn new_addition(item: Export) -> Change<'tcx> {
        Change::Unary(UnaryChange::Addition(item))
    }

    /// Construct a new removal-change for the given export.
    fn new_removal(item: Export) -> Change<'tcx> {
        Change::Unary(UnaryChange::Removal(item))
    }

    /// Construct a new binary change for the given exports.
    fn new_binary(name: Symbol, span: Span, output: bool) -> Change<'tcx> {
        Change::Binary(BinaryChange::new(name, span, output))
    }

    /// Add a change type to a given binary change.
    fn add(&mut self, type_: BinaryChangeType<'tcx>, span: Option<Span>) {
        match *self {
            Change::Unary(_) => panic!("can't add binary change types to unary change"),
            Change::Binary(ref mut b) => b.add(type_, span),
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
pub struct ChangeSet<'tcx> {
    /// The currently recorded changes.
    changes: HashMap<ChangeKey, Change<'tcx>>,
    /// The mapping of spans to changes, for ordering purposes.
    spans: BTreeMap<Span, ChangeKey>,
    /// The most severe change category already recorded.
    max: ChangeCategory,
}

impl<'tcx> ChangeSet<'tcx> {
    /// Add a new addition-change for the given export.
    pub fn new_addition(&mut self, item: Export) {
        self.new_unary_change(Change::new_addition(item), ChangeKey::NewKey(item.def.def_id()));
    }

    /// Add a new removal-change for the given export.
    pub fn new_removal(&mut self, item: Export) {
        self.new_unary_change(Change::new_removal(item), ChangeKey::NewKey(item.def.def_id()));
    }

    /// Add a new (unary) change for the given key.
    fn new_unary_change(&mut self, change: Change<'tcx>, key: ChangeKey) {
        let cat = change.to_category();

        if cat > self.max {
            self.max = cat.clone();
        }

        self.spans.insert(*change.span(), key.clone());
        self.changes.insert(key, change);
    }

    /// Add a new binary change entry for the given exports.
    pub fn new_binary(&mut self, old_did: DefId, name: Symbol, span: Span, output: bool) {
        let key = ChangeKey::OldKey(old_did);
        let change = Change::new_binary(name, span, output);

        self.spans.insert(*change.span(), key.clone());
        self.changes.insert(key, change);
    }

    /// Add a new binary change to an already existing entry.
    pub fn add_binary(&mut self, type_: BinaryChangeType<'tcx>, old: DefId, span: Option<Span>) {
        let key = ChangeKey::OldKey(old);
        let cat = type_.to_category();

        if cat > self.max {
            self.max = cat.clone();
        }

        self.changes.get_mut(&key).unwrap().add(type_, span);
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

    pub type BinaryChange_ = (Span_, Span_);

    fn build_binary_change(t: BinaryChangeType, s1: Span, s2: Span) -> BinaryChange {
        let mut interner = Interner::new();
        let mut change = BinaryChange::new(interner.intern("test"), s2, true);
        change.add(t, Some(s1));

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
            let ch1 = build_binary_change(Unknown, c1.0.inner(), c1.1.inner());
            let ch2 = build_binary_change(Unknown, c2.0.inner(), c2.1.inner());
            let ch3 = build_binary_change(Unknown, c3.0.inner(), c3.1.inner());

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

            let max = changes.iter().map(|_| Unknown.to_category()).max().unwrap_or(Patch);

            for &(ref span1, ref span2) in changes.iter() {
                let change =
                    build_binary_change(Unknown,
                                        span1.clone().inner(),
                                        span2.clone().inner());
                let did = DefId {
                    krate: LOCAL_CRATE,
                    index: CRATE_DEF_INDEX,
                };
                let key = ChangeKey::OldKey(did);
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
            let s1 = c1.0.inner();
            let s2 = c2.0.inner();

            if s1 != s2 {
                let ch1 = build_binary_change(Unknown, c1.1.inner(), s1);
                let ch2 = build_binary_change(Unknown, c2.1.inner(), s2);

                ch1 != ch2
            } else {
                true
            }
        }
    }
}
