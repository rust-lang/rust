//! Machinery for hygienic macros.
//!
//! Inspired by Matthew Flatt et al., “Macros That Work Together: Compile-Time Bindings, Partial
//! Expansion, and Definition Contexts,” *Journal of Functional Programming* 22, no. 2
//! (March 1, 2012): 181–216, <https://doi.org/10.1017/S0956796812000093>.

// Hygiene data is stored in a global variable and accessed via TLS, which
// means that accesses are somewhat expensive. (`HygieneData::with`
// encapsulates a single access.) Therefore, on hot code paths it is worth
// ensuring that multiple HygieneData accesses are combined into a single
// `HygieneData::with`.
//
// This explains why `HygieneData`, `SyntaxContext` and `ExpnId` have interfaces
// with a certain amount of redundancy in them. For example,
// `SyntaxContext::outer_expn_data` combines `SyntaxContext::outer` and
// `ExpnId::expn_data` so that two `HygieneData` accesses can be performed within
// a single `HygieneData::with` call.
//
// It also explains why many functions appear in `HygieneData` and again in
// `SyntaxContext` or `ExpnId`. For example, `HygieneData::outer` and
// `SyntaxContext::outer` do the same thing, but the former is for use within a
// `HygieneData::with` call while the latter is for use outside such a call.
// When modifying this file it is important to understand this distinction,
// because getting it wrong can lead to nested `HygieneData::with` calls that
// trigger runtime aborts. (Fortunately these are obvious and easy to fix.)

use std::hash::Hash;
use std::sync::Arc;
use std::{fmt, iter, mem};

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::{HashStable, HashingControls, StableHasher};
use rustc_data_structures::sync::Lock;
use rustc_data_structures::unhash::UnhashMap;
use rustc_hashes::Hash64;
use rustc_index::IndexVec;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use tracing::{debug, trace};

use crate::def_id::{CRATE_DEF_ID, CrateNum, DefId, LOCAL_CRATE, StableCrateId};
use crate::edition::Edition;
use crate::source_map::SourceMap;
use crate::symbol::{Symbol, kw, sym};
use crate::{DUMMY_SP, HashStableContext, Span, SpanDecoder, SpanEncoder, with_session_globals};

/// A `SyntaxContext` represents a chain of pairs `(ExpnId, Transparency)` named "marks".
///
/// See <https://rustc-dev-guide.rust-lang.org/macro-expansion.html> for more explanation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyntaxContext(u32);

// To ensure correctness of incremental compilation,
// `SyntaxContext` must not implement `Ord` or `PartialOrd`.
// See https://github.com/rust-lang/rust/issues/90317.
impl !Ord for SyntaxContext {}
impl !PartialOrd for SyntaxContext {}

/// If this part of two syntax contexts is equal, then the whole syntax contexts should be equal.
/// The other fields are only for caching.
pub type SyntaxContextKey = (SyntaxContext, ExpnId, Transparency);

#[derive(Clone, Copy, Debug)]
struct SyntaxContextData {
    /// The last macro expansion in the chain.
    /// (Here we say the most deeply nested macro expansion is the "outermost" expansion.)
    outer_expn: ExpnId,
    /// Transparency of the last macro expansion
    outer_transparency: Transparency,
    parent: SyntaxContext,
    /// This context, but with all transparent and semi-opaque expansions filtered away.
    opaque: SyntaxContext,
    /// This context, but with all transparent expansions filtered away.
    opaque_and_semiopaque: SyntaxContext,
    /// Name of the crate to which `$crate` with this context would resolve.
    dollar_crate_name: Symbol,
}

impl SyntaxContextData {
    fn root() -> SyntaxContextData {
        SyntaxContextData {
            outer_expn: ExpnId::root(),
            outer_transparency: Transparency::Opaque,
            parent: SyntaxContext::root(),
            opaque: SyntaxContext::root(),
            opaque_and_semiopaque: SyntaxContext::root(),
            dollar_crate_name: kw::DollarCrate,
        }
    }

    fn key(&self) -> SyntaxContextKey {
        (self.parent, self.outer_expn, self.outer_transparency)
    }
}

rustc_index::newtype_index! {
    /// A unique ID associated with a macro invocation and expansion.
    #[orderable]
    pub struct ExpnIndex {}
}

/// A unique ID associated with a macro invocation and expansion.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpnId {
    pub krate: CrateNum,
    pub local_id: ExpnIndex,
}

impl fmt::Debug for ExpnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Generate crate_::{{expn_}}.
        write!(f, "{:?}::{{{{expn{}}}}}", self.krate, self.local_id.as_u32())
    }
}

rustc_index::newtype_index! {
    /// A unique ID associated with a macro invocation and expansion.
    #[debug_format = "expn{}"]
    pub struct LocalExpnId {}
}

// To ensure correctness of incremental compilation,
// `LocalExpnId` must not implement `Ord` or `PartialOrd`.
// See https://github.com/rust-lang/rust/issues/90317.
impl !Ord for LocalExpnId {}
impl !PartialOrd for LocalExpnId {}

/// Assert that the provided `HashStableContext` is configured with the 'default'
/// `HashingControls`. We should always have bailed out before getting to here
/// with a non-default mode. With this check in place, we can avoid the need
/// to maintain separate versions of `ExpnData` hashes for each permutation
/// of `HashingControls` settings.
fn assert_default_hashing_controls(ctx: &impl HashStableContext, msg: &str) {
    match ctx.hashing_controls() {
        // Note that we require that `hash_spans` be set according to the global
        // `-Z incremental-ignore-spans` option. Normally, this option is disabled,
        // which will cause us to require that this method always be called with `Span` hashing
        // enabled.
        //
        // Span hashing can also be disabled without `-Z incremental-ignore-spans`.
        // This is the case for instance when building a hash for name mangling.
        // Such configuration must not be used for metadata.
        HashingControls { hash_spans }
            if hash_spans != ctx.unstable_opts_incremental_ignore_spans() => {}
        other => panic!("Attempted hashing of {msg} with non-default HashingControls: {other:?}"),
    }
}

/// A unique hash value associated to an expansion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Encodable, Decodable, HashStable_Generic)]
pub struct ExpnHash(Fingerprint);

impl ExpnHash {
    /// Returns the [StableCrateId] identifying the crate this [ExpnHash]
    /// originates from.
    #[inline]
    pub fn stable_crate_id(self) -> StableCrateId {
        StableCrateId(self.0.split().0)
    }

    /// Returns the crate-local part of the [ExpnHash].
    ///
    /// Used for assertions.
    #[inline]
    pub fn local_hash(self) -> Hash64 {
        self.0.split().1
    }

    #[inline]
    pub fn is_root(self) -> bool {
        self.0 == Fingerprint::ZERO
    }

    /// Builds a new [ExpnHash] with the given [StableCrateId] and
    /// `local_hash`, where `local_hash` must be unique within its crate.
    fn new(stable_crate_id: StableCrateId, local_hash: Hash64) -> ExpnHash {
        ExpnHash(Fingerprint::new(stable_crate_id.0, local_hash))
    }
}

/// A property of a macro expansion that determines how identifiers
/// produced by that expansion are resolved.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub enum Transparency {
    /// Identifier produced by a transparent expansion is always resolved at call-site.
    /// Call-site spans in procedural macros, hygiene opt-out in `macro` should use this.
    Transparent,
    /// Identifier produced by a semi-opaque expansion may be resolved
    /// either at call-site or at definition-site.
    /// If it's a local variable, label or `$crate` then it's resolved at def-site.
    /// Otherwise it's resolved at call-site.
    /// `macro_rules` macros behave like this, built-in macros currently behave like this too,
    /// but that's an implementation detail.
    SemiOpaque,
    /// Identifier produced by an opaque expansion is always resolved at definition-site.
    /// Def-site spans in procedural macros, identifiers from `macro` by default use this.
    Opaque,
}

impl Transparency {
    pub fn fallback(macro_rules: bool) -> Self {
        if macro_rules { Transparency::SemiOpaque } else { Transparency::Opaque }
    }
}

impl LocalExpnId {
    /// The ID of the theoretical expansion that generates freshly parsed, unexpanded AST.
    pub const ROOT: LocalExpnId = LocalExpnId::ZERO;

    #[inline]
    fn from_raw(idx: ExpnIndex) -> LocalExpnId {
        LocalExpnId::from_u32(idx.as_u32())
    }

    #[inline]
    pub fn as_raw(self) -> ExpnIndex {
        ExpnIndex::from_u32(self.as_u32())
    }

    pub fn fresh_empty() -> LocalExpnId {
        HygieneData::with(|data| {
            let expn_id = data.local_expn_data.push(None);
            let _eid = data.local_expn_hashes.push(ExpnHash(Fingerprint::ZERO));
            debug_assert_eq!(expn_id, _eid);
            expn_id
        })
    }

    pub fn fresh(mut expn_data: ExpnData, ctx: impl HashStableContext) -> LocalExpnId {
        debug_assert_eq!(expn_data.parent.krate, LOCAL_CRATE);
        let expn_hash = update_disambiguator(&mut expn_data, ctx);
        HygieneData::with(|data| {
            let expn_id = data.local_expn_data.push(Some(expn_data));
            let _eid = data.local_expn_hashes.push(expn_hash);
            debug_assert_eq!(expn_id, _eid);
            let _old_id = data.expn_hash_to_expn_id.insert(expn_hash, expn_id.to_expn_id());
            debug_assert!(_old_id.is_none());
            expn_id
        })
    }

    #[inline]
    pub fn expn_data(self) -> ExpnData {
        HygieneData::with(|data| data.local_expn_data(self).clone())
    }

    #[inline]
    pub fn to_expn_id(self) -> ExpnId {
        ExpnId { krate: LOCAL_CRATE, local_id: self.as_raw() }
    }

    #[inline]
    pub fn set_expn_data(self, mut expn_data: ExpnData, ctx: impl HashStableContext) {
        debug_assert_eq!(expn_data.parent.krate, LOCAL_CRATE);
        let expn_hash = update_disambiguator(&mut expn_data, ctx);
        HygieneData::with(|data| {
            let old_expn_data = &mut data.local_expn_data[self];
            assert!(old_expn_data.is_none(), "expansion data is reset for an expansion ID");
            *old_expn_data = Some(expn_data);
            debug_assert_eq!(data.local_expn_hashes[self].0, Fingerprint::ZERO);
            data.local_expn_hashes[self] = expn_hash;
            let _old_id = data.expn_hash_to_expn_id.insert(expn_hash, self.to_expn_id());
            debug_assert!(_old_id.is_none());
        });
    }

    #[inline]
    pub fn is_descendant_of(self, ancestor: LocalExpnId) -> bool {
        self.to_expn_id().is_descendant_of(ancestor.to_expn_id())
    }

    /// Returns span for the macro which originally caused this expansion to happen.
    ///
    /// Stops backtracing at include! boundary.
    #[inline]
    pub fn expansion_cause(self) -> Option<Span> {
        self.to_expn_id().expansion_cause()
    }
}

impl ExpnId {
    /// The ID of the theoretical expansion that generates freshly parsed, unexpanded AST.
    /// Invariant: we do not create any ExpnId with local_id == 0 and krate != 0.
    pub const fn root() -> ExpnId {
        ExpnId { krate: LOCAL_CRATE, local_id: ExpnIndex::ZERO }
    }

    #[inline]
    pub fn expn_hash(self) -> ExpnHash {
        HygieneData::with(|data| data.expn_hash(self))
    }

    #[inline]
    pub fn from_hash(hash: ExpnHash) -> Option<ExpnId> {
        HygieneData::with(|data| data.expn_hash_to_expn_id.get(&hash).copied())
    }

    #[inline]
    pub fn as_local(self) -> Option<LocalExpnId> {
        if self.krate == LOCAL_CRATE { Some(LocalExpnId::from_raw(self.local_id)) } else { None }
    }

    #[inline]
    #[track_caller]
    pub fn expect_local(self) -> LocalExpnId {
        self.as_local().unwrap()
    }

    #[inline]
    pub fn expn_data(self) -> ExpnData {
        HygieneData::with(|data| data.expn_data(self).clone())
    }

    #[inline]
    pub fn is_descendant_of(self, ancestor: ExpnId) -> bool {
        // a few "fast path" cases to avoid locking HygieneData
        if ancestor == ExpnId::root() || ancestor == self {
            return true;
        }
        if ancestor.krate != self.krate {
            return false;
        }
        HygieneData::with(|data| data.is_descendant_of(self, ancestor))
    }

    /// `expn_id.outer_expn_is_descendant_of(ctxt)` is equivalent to but faster than
    /// `expn_id.is_descendant_of(ctxt.outer_expn())`.
    #[inline]
    pub fn outer_expn_is_descendant_of(self, ctxt: SyntaxContext) -> bool {
        HygieneData::with(|data| data.is_descendant_of(self, data.outer_expn(ctxt)))
    }

    /// Returns span for the macro which originally caused this expansion to happen.
    ///
    /// Stops backtracing at include! boundary.
    pub fn expansion_cause(mut self) -> Option<Span> {
        let mut last_macro = None;
        loop {
            // Fast path to avoid locking.
            if self == ExpnId::root() {
                break;
            }
            let expn_data = self.expn_data();
            // Stop going up the backtrace once include! is encountered
            if expn_data.kind == ExpnKind::Macro(MacroKind::Bang, sym::include) {
                break;
            }
            self = expn_data.call_site.ctxt().outer_expn();
            last_macro = Some(expn_data.call_site);
        }
        last_macro
    }
}

#[derive(Debug)]
pub(crate) struct HygieneData {
    /// Each expansion should have an associated expansion data, but sometimes there's a delay
    /// between creation of an expansion ID and obtaining its data (e.g. macros are collected
    /// first and then resolved later), so we use an `Option` here.
    local_expn_data: IndexVec<LocalExpnId, Option<ExpnData>>,
    local_expn_hashes: IndexVec<LocalExpnId, ExpnHash>,
    /// Data and hash information from external crates. We may eventually want to remove these
    /// maps, and fetch the information directly from the other crate's metadata like DefIds do.
    foreign_expn_data: FxHashMap<ExpnId, ExpnData>,
    foreign_expn_hashes: FxHashMap<ExpnId, ExpnHash>,
    expn_hash_to_expn_id: UnhashMap<ExpnHash, ExpnId>,
    syntax_context_data: Vec<SyntaxContextData>,
    syntax_context_map: FxHashMap<SyntaxContextKey, SyntaxContext>,
    /// Maps the `local_hash` of an `ExpnData` to the next disambiguator value.
    /// This is used by `update_disambiguator` to keep track of which `ExpnData`s
    /// would have collisions without a disambiguator.
    /// The keys of this map are always computed with `ExpnData.disambiguator`
    /// set to 0.
    expn_data_disambiguators: UnhashMap<Hash64, u32>,
}

impl HygieneData {
    pub(crate) fn new(edition: Edition) -> Self {
        let root_data = ExpnData::default(
            ExpnKind::Root,
            DUMMY_SP,
            edition,
            Some(CRATE_DEF_ID.to_def_id()),
            None,
        );

        let root_ctxt_data = SyntaxContextData::root();
        HygieneData {
            local_expn_data: IndexVec::from_elem_n(Some(root_data), 1),
            local_expn_hashes: IndexVec::from_elem_n(ExpnHash(Fingerprint::ZERO), 1),
            foreign_expn_data: FxHashMap::default(),
            foreign_expn_hashes: FxHashMap::default(),
            expn_hash_to_expn_id: iter::once((ExpnHash(Fingerprint::ZERO), ExpnId::root()))
                .collect(),
            syntax_context_data: vec![root_ctxt_data],
            syntax_context_map: iter::once((root_ctxt_data.key(), SyntaxContext(0))).collect(),
            expn_data_disambiguators: UnhashMap::default(),
        }
    }

    #[inline]
    fn with<R>(f: impl FnOnce(&mut HygieneData) -> R) -> R {
        with_session_globals(|session_globals| f(&mut session_globals.hygiene_data.borrow_mut()))
    }

    #[inline]
    fn expn_hash(&self, expn_id: ExpnId) -> ExpnHash {
        match expn_id.as_local() {
            Some(expn_id) => self.local_expn_hashes[expn_id],
            None => self.foreign_expn_hashes[&expn_id],
        }
    }

    #[inline]
    fn local_expn_data(&self, expn_id: LocalExpnId) -> &ExpnData {
        self.local_expn_data[expn_id].as_ref().expect("no expansion data for an expansion ID")
    }

    fn expn_data(&self, expn_id: ExpnId) -> &ExpnData {
        if let Some(expn_id) = expn_id.as_local() {
            self.local_expn_data[expn_id].as_ref().expect("no expansion data for an expansion ID")
        } else {
            &self.foreign_expn_data[&expn_id]
        }
    }

    fn is_descendant_of(&self, mut expn_id: ExpnId, ancestor: ExpnId) -> bool {
        // a couple "fast path" cases to avoid traversing parents in the loop below
        if ancestor == ExpnId::root() {
            return true;
        }
        if expn_id.krate != ancestor.krate {
            return false;
        }
        loop {
            if expn_id == ancestor {
                return true;
            }
            if expn_id == ExpnId::root() {
                return false;
            }
            expn_id = self.expn_data(expn_id).parent;
        }
    }

    #[inline]
    fn normalize_to_macros_2_0(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].opaque
    }

    #[inline]
    fn normalize_to_macro_rules(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].opaque_and_semiopaque
    }

    /// See [`SyntaxContextData::outer_expn`]
    #[inline]
    fn outer_expn(&self, ctxt: SyntaxContext) -> ExpnId {
        self.syntax_context_data[ctxt.0 as usize].outer_expn
    }

    /// The last macro expansion and its Transparency
    #[inline]
    fn outer_mark(&self, ctxt: SyntaxContext) -> (ExpnId, Transparency) {
        let data = &self.syntax_context_data[ctxt.0 as usize];
        (data.outer_expn, data.outer_transparency)
    }

    #[inline]
    fn parent_ctxt(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].parent
    }

    fn remove_mark(&self, ctxt: &mut SyntaxContext) -> (ExpnId, Transparency) {
        let outer_mark = self.outer_mark(*ctxt);
        *ctxt = self.parent_ctxt(*ctxt);
        outer_mark
    }

    fn marks(&self, mut ctxt: SyntaxContext) -> Vec<(ExpnId, Transparency)> {
        let mut marks = Vec::new();
        while !ctxt.is_root() {
            debug!("marks: getting parent of {:?}", ctxt);
            marks.push(self.outer_mark(ctxt));
            ctxt = self.parent_ctxt(ctxt);
        }
        marks.reverse();
        marks
    }

    fn walk_chain(&self, mut span: Span, to: SyntaxContext) -> Span {
        let orig_span = span;
        debug!("walk_chain({:?}, {:?})", span, to);
        debug!("walk_chain: span ctxt = {:?}", span.ctxt());
        while span.ctxt() != to && span.from_expansion() {
            let outer_expn = self.outer_expn(span.ctxt());
            debug!("walk_chain({:?}): outer_expn={:?}", span, outer_expn);
            let expn_data = self.expn_data(outer_expn);
            debug!("walk_chain({:?}): expn_data={:?}", span, expn_data);
            span = expn_data.call_site;
        }
        debug!("walk_chain: for span {:?} >>> return span = {:?}", orig_span, span);
        span
    }

    fn walk_chain_collapsed(&self, mut span: Span, to: Span) -> Span {
        let orig_span = span;
        let mut ret_span = span;
        debug!("walk_chain_collapsed({:?}, {:?})", span, to);
        debug!("walk_chain_collapsed: span ctxt = {:?}", span.ctxt());
        while let ctxt = span.ctxt()
            && !ctxt.is_root()
            && ctxt != to.ctxt()
        {
            let outer_expn = self.outer_expn(ctxt);
            debug!("walk_chain_collapsed({:?}): outer_expn={:?}", span, outer_expn);
            let expn_data = self.expn_data(outer_expn);
            debug!("walk_chain_collapsed({:?}): expn_data={:?}", span, expn_data);
            span = expn_data.call_site;
            if expn_data.collapse_debuginfo {
                ret_span = span;
            }
        }
        debug!("walk_chain_collapsed: for span {:?} >>> return span = {:?}", orig_span, ret_span);
        ret_span
    }

    fn adjust(&self, ctxt: &mut SyntaxContext, expn_id: ExpnId) -> Option<ExpnId> {
        let mut scope = None;
        while !self.is_descendant_of(expn_id, self.outer_expn(*ctxt)) {
            scope = Some(self.remove_mark(ctxt).0);
        }
        scope
    }

    fn apply_mark(
        &mut self,
        ctxt: SyntaxContext,
        expn_id: ExpnId,
        transparency: Transparency,
    ) -> SyntaxContext {
        assert_ne!(expn_id, ExpnId::root());
        if transparency == Transparency::Opaque {
            return self.alloc_ctxt(ctxt, expn_id, transparency);
        }

        let call_site_ctxt = self.expn_data(expn_id).call_site.ctxt();
        let mut call_site_ctxt = if transparency == Transparency::SemiOpaque {
            self.normalize_to_macros_2_0(call_site_ctxt)
        } else {
            self.normalize_to_macro_rules(call_site_ctxt)
        };

        if call_site_ctxt.is_root() {
            return self.alloc_ctxt(ctxt, expn_id, transparency);
        }

        // Otherwise, `expn_id` is a macros 1.0 definition and the call site is in a
        // macros 2.0 expansion, i.e., a macros 1.0 invocation is in a macros 2.0 definition.
        //
        // In this case, the tokens from the macros 1.0 definition inherit the hygiene
        // at their invocation. That is, we pretend that the macros 1.0 definition
        // was defined at its invocation (i.e., inside the macros 2.0 definition)
        // so that the macros 2.0 definition remains hygienic.
        //
        // See the example at `test/ui/hygiene/legacy_interaction.rs`.
        for (expn_id, transparency) in self.marks(ctxt) {
            call_site_ctxt = self.alloc_ctxt(call_site_ctxt, expn_id, transparency);
        }
        self.alloc_ctxt(call_site_ctxt, expn_id, transparency)
    }

    /// Allocate a new context with the given key, or retrieve it from cache if the given key
    /// already exists. The auxiliary fields are calculated from the key.
    fn alloc_ctxt(
        &mut self,
        parent: SyntaxContext,
        expn_id: ExpnId,
        transparency: Transparency,
    ) -> SyntaxContext {
        // Look into the cache first.
        let key = (parent, expn_id, transparency);
        if let Some(ctxt) = self.syntax_context_map.get(&key) {
            return *ctxt;
        }

        // Reserve a new syntax context.
        // The inserted dummy data can only be potentially accessed by nested `alloc_ctxt` calls,
        // the assert below ensures that it doesn't happen.
        let ctxt = SyntaxContext::from_usize(self.syntax_context_data.len());
        self.syntax_context_data
            .push(SyntaxContextData { dollar_crate_name: sym::dummy, ..SyntaxContextData::root() });
        self.syntax_context_map.insert(key, ctxt);

        // Opaque and semi-opaque versions of the parent. Note that they may be equal to the
        // parent itself. E.g. `parent_opaque` == `parent` if the expn chain contains only opaques,
        // and `parent_opaque_and_semiopaque` == `parent` if the expn contains only (semi-)opaques.
        let parent_data = &self.syntax_context_data[parent.0 as usize];
        assert_ne!(parent_data.dollar_crate_name, sym::dummy);
        let parent_opaque = parent_data.opaque;
        let parent_opaque_and_semiopaque = parent_data.opaque_and_semiopaque;

        // Evaluate opaque and semi-opaque versions of the new syntax context.
        let (opaque, opaque_and_semiopaque) = match transparency {
            Transparency::Transparent => (parent_opaque, parent_opaque_and_semiopaque),
            Transparency::SemiOpaque => (
                parent_opaque,
                // Will be the same as `ctxt` if the expn chain contains only (semi-)opaques.
                self.alloc_ctxt(parent_opaque_and_semiopaque, expn_id, transparency),
            ),
            Transparency::Opaque => (
                // Will be the same as `ctxt` if the expn chain contains only opaques.
                self.alloc_ctxt(parent_opaque, expn_id, transparency),
                // Will be the same as `ctxt` if the expn chain contains only (semi-)opaques.
                self.alloc_ctxt(parent_opaque_and_semiopaque, expn_id, transparency),
            ),
        };

        // Fill the full data, now that we have it.
        self.syntax_context_data[ctxt.as_u32() as usize] = SyntaxContextData {
            outer_expn: expn_id,
            outer_transparency: transparency,
            parent,
            opaque,
            opaque_and_semiopaque,
            dollar_crate_name: kw::DollarCrate,
        };
        ctxt
    }
}

pub fn walk_chain(span: Span, to: SyntaxContext) -> Span {
    HygieneData::with(|data| data.walk_chain(span, to))
}

/// In order to have good line stepping behavior in debugger, for the given span we return its
/// outermost macro call site that still has a `#[collapse_debuginfo(yes)]` property on it.
/// We also stop walking call sites at the function body level because no line stepping can occur
/// at the level above that.
/// The returned span can then be used in emitted debuginfo.
pub fn walk_chain_collapsed(span: Span, to: Span) -> Span {
    HygieneData::with(|data| data.walk_chain_collapsed(span, to))
}

pub fn update_dollar_crate_names(mut get_name: impl FnMut(SyntaxContext) -> Symbol) {
    // The new contexts that need updating are at the end of the list and have `$crate` as a name.
    let mut to_update = vec![];
    HygieneData::with(|data| {
        for (idx, scdata) in data.syntax_context_data.iter().enumerate().rev() {
            if scdata.dollar_crate_name == kw::DollarCrate {
                to_update.push((idx, kw::DollarCrate));
            } else {
                break;
            }
        }
    });
    // The callback must be called from outside of the `HygieneData` lock,
    // since it will try to acquire it too.
    for (idx, name) in &mut to_update {
        *name = get_name(SyntaxContext::from_usize(*idx));
    }
    HygieneData::with(|data| {
        for (idx, name) in to_update {
            data.syntax_context_data[idx].dollar_crate_name = name;
        }
    })
}

pub fn debug_hygiene_data(verbose: bool) -> String {
    HygieneData::with(|data| {
        if verbose {
            format!("{data:#?}")
        } else {
            let mut s = String::from("Expansions:");
            let mut debug_expn_data = |(id, expn_data): (&ExpnId, &ExpnData)| {
                s.push_str(&format!(
                    "\n{:?}: parent: {:?}, call_site_ctxt: {:?}, def_site_ctxt: {:?}, kind: {:?}",
                    id,
                    expn_data.parent,
                    expn_data.call_site.ctxt(),
                    expn_data.def_site.ctxt(),
                    expn_data.kind,
                ))
            };
            data.local_expn_data.iter_enumerated().for_each(|(id, expn_data)| {
                let expn_data = expn_data.as_ref().expect("no expansion data for an expansion ID");
                debug_expn_data((&id.to_expn_id(), expn_data))
            });

            // Sort the hash map for more reproducible output.
            // Because of this, it is fine to rely on the unstable iteration order of the map.
            #[allow(rustc::potential_query_instability)]
            let mut foreign_expn_data: Vec<_> = data.foreign_expn_data.iter().collect();
            foreign_expn_data.sort_by_key(|(id, _)| (id.krate, id.local_id));
            foreign_expn_data.into_iter().for_each(debug_expn_data);
            s.push_str("\n\nSyntaxContexts:");
            data.syntax_context_data.iter().enumerate().for_each(|(id, ctxt)| {
                s.push_str(&format!(
                    "\n#{}: parent: {:?}, outer_mark: ({:?}, {:?})",
                    id, ctxt.parent, ctxt.outer_expn, ctxt.outer_transparency,
                ));
            });
            s
        }
    })
}

impl SyntaxContext {
    #[inline]
    pub const fn root() -> Self {
        SyntaxContext(0)
    }

    #[inline]
    pub const fn is_root(self) -> bool {
        self.0 == SyntaxContext::root().as_u32()
    }

    #[inline]
    pub(crate) const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub(crate) const fn from_u32(raw: u32) -> SyntaxContext {
        SyntaxContext(raw)
    }

    #[inline]
    pub(crate) const fn from_u16(raw: u16) -> SyntaxContext {
        SyntaxContext(raw as u32)
    }

    #[inline]
    fn from_usize(raw: usize) -> SyntaxContext {
        SyntaxContext(u32::try_from(raw).unwrap())
    }

    /// Extend a syntax context with a given expansion and transparency.
    #[inline]
    pub fn apply_mark(self, expn_id: ExpnId, transparency: Transparency) -> SyntaxContext {
        HygieneData::with(|data| data.apply_mark(self, expn_id, transparency))
    }

    /// Pulls a single mark off of the syntax context. This effectively moves the
    /// context up one macro definition level. That is, if we have a nested macro
    /// definition as follows:
    ///
    /// ```ignore (illustrative)
    /// macro_rules! f {
    ///    macro_rules! g {
    ///        ...
    ///    }
    /// }
    /// ```
    ///
    /// and we have a SyntaxContext that is referring to something declared by an invocation
    /// of g (call it g1), calling remove_mark will result in the SyntaxContext for the
    /// invocation of f that created g1.
    /// Returns the mark that was removed.
    #[inline]
    pub fn remove_mark(&mut self) -> ExpnId {
        HygieneData::with(|data| data.remove_mark(self).0)
    }

    #[inline]
    pub fn marks(self) -> Vec<(ExpnId, Transparency)> {
        HygieneData::with(|data| data.marks(self))
    }

    /// Adjust this context for resolution in a scope created by the given expansion.
    /// For example, consider the following three resolutions of `f`:
    ///
    /// ```rust
    /// #![feature(decl_macro)]
    /// mod foo {
    ///     pub fn f() {} // `f`'s `SyntaxContext` is empty.
    /// }
    /// m!(f);
    /// macro m($f:ident) {
    ///     mod bar {
    ///         pub fn f() {} // `f`'s `SyntaxContext` has a single `ExpnId` from `m`.
    ///         pub fn $f() {} // `$f`'s `SyntaxContext` is empty.
    ///     }
    ///     foo::f(); // `f`'s `SyntaxContext` has a single `ExpnId` from `m`
    ///     //^ Since `mod foo` is outside this expansion, `adjust` removes the mark from `f`,
    ///     //| and it resolves to `::foo::f`.
    ///     bar::f(); // `f`'s `SyntaxContext` has a single `ExpnId` from `m`
    ///     //^ Since `mod bar` not outside this expansion, `adjust` does not change `f`,
    ///     //| and it resolves to `::bar::f`.
    ///     bar::$f(); // `f`'s `SyntaxContext` is empty.
    ///     //^ Since `mod bar` is not outside this expansion, `adjust` does not change `$f`,
    ///     //| and it resolves to `::bar::$f`.
    /// }
    /// ```
    /// This returns the expansion whose definition scope we use to privacy check the resolution,
    /// or `None` if we privacy check as usual (i.e., not w.r.t. a macro definition scope).
    #[inline]
    pub fn adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        HygieneData::with(|data| data.adjust(self, expn_id))
    }

    /// Like `SyntaxContext::adjust`, but also normalizes `self` to macros 2.0.
    #[inline]
    pub(crate) fn normalize_to_macros_2_0_and_adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        HygieneData::with(|data| {
            *self = data.normalize_to_macros_2_0(*self);
            data.adjust(self, expn_id)
        })
    }

    /// Adjust this context for resolution in a scope created by the given expansion
    /// via a glob import with the given `SyntaxContext`.
    /// For example:
    ///
    /// ```compile_fail,E0425
    /// #![feature(decl_macro)]
    /// m!(f);
    /// macro m($i:ident) {
    ///     mod foo {
    ///         pub fn f() {} // `f`'s `SyntaxContext` has a single `ExpnId` from `m`.
    ///         pub fn $i() {} // `$i`'s `SyntaxContext` is empty.
    ///     }
    ///     n!(f);
    ///     macro n($j:ident) {
    ///         use foo::*;
    ///         f(); // `f`'s `SyntaxContext` has a mark from `m` and a mark from `n`
    ///         //^ `glob_adjust` removes the mark from `n`, so this resolves to `foo::f`.
    ///         $i(); // `$i`'s `SyntaxContext` has a mark from `n`
    ///         //^ `glob_adjust` removes the mark from `n`, so this resolves to `foo::$i`.
    ///         $j(); // `$j`'s `SyntaxContext` has a mark from `m`
    ///         //^ This cannot be glob-adjusted, so this is a resolution error.
    ///     }
    /// }
    /// ```
    /// This returns `None` if the context cannot be glob-adjusted.
    /// Otherwise, it returns the scope to use when privacy checking (see `adjust` for details).
    pub(crate) fn glob_adjust(
        &mut self,
        expn_id: ExpnId,
        glob_span: Span,
    ) -> Option<Option<ExpnId>> {
        HygieneData::with(|data| {
            let mut scope = None;
            let mut glob_ctxt = data.normalize_to_macros_2_0(glob_span.ctxt());
            while !data.is_descendant_of(expn_id, data.outer_expn(glob_ctxt)) {
                scope = Some(data.remove_mark(&mut glob_ctxt).0);
                if data.remove_mark(self).0 != scope.unwrap() {
                    return None;
                }
            }
            if data.adjust(self, expn_id).is_some() {
                return None;
            }
            Some(scope)
        })
    }

    /// Undo `glob_adjust` if possible:
    ///
    /// ```ignore (illustrative)
    /// if let Some(privacy_checking_scope) = self.reverse_glob_adjust(expansion, glob_ctxt) {
    ///     assert!(self.glob_adjust(expansion, glob_ctxt) == Some(privacy_checking_scope));
    /// }
    /// ```
    pub(crate) fn reverse_glob_adjust(
        &mut self,
        expn_id: ExpnId,
        glob_span: Span,
    ) -> Option<Option<ExpnId>> {
        HygieneData::with(|data| {
            if data.adjust(self, expn_id).is_some() {
                return None;
            }

            let mut glob_ctxt = data.normalize_to_macros_2_0(glob_span.ctxt());
            let mut marks = Vec::new();
            while !data.is_descendant_of(expn_id, data.outer_expn(glob_ctxt)) {
                marks.push(data.remove_mark(&mut glob_ctxt));
            }

            let scope = marks.last().map(|mark| mark.0);
            while let Some((expn_id, transparency)) = marks.pop() {
                *self = data.apply_mark(*self, expn_id, transparency);
            }
            Some(scope)
        })
    }

    pub fn hygienic_eq(self, other: SyntaxContext, expn_id: ExpnId) -> bool {
        HygieneData::with(|data| {
            let mut self_normalized = data.normalize_to_macros_2_0(self);
            data.adjust(&mut self_normalized, expn_id);
            self_normalized == data.normalize_to_macros_2_0(other)
        })
    }

    #[inline]
    pub fn normalize_to_macros_2_0(self) -> SyntaxContext {
        HygieneData::with(|data| data.normalize_to_macros_2_0(self))
    }

    #[inline]
    pub fn normalize_to_macro_rules(self) -> SyntaxContext {
        HygieneData::with(|data| data.normalize_to_macro_rules(self))
    }

    /// See [`SyntaxContextData::outer_expn`]
    #[inline]
    pub fn outer_expn(self) -> ExpnId {
        HygieneData::with(|data| data.outer_expn(self))
    }

    /// `ctxt.outer_expn_data()` is equivalent to but faster than
    /// `ctxt.outer_expn().expn_data()`.
    #[inline]
    pub fn outer_expn_data(self) -> ExpnData {
        HygieneData::with(|data| data.expn_data(data.outer_expn(self)).clone())
    }

    /// See [`HygieneData::outer_mark`]
    #[inline]
    fn outer_mark(self) -> (ExpnId, Transparency) {
        HygieneData::with(|data| data.outer_mark(self))
    }

    #[inline]
    pub(crate) fn dollar_crate_name(self) -> Symbol {
        HygieneData::with(|data| data.syntax_context_data[self.0 as usize].dollar_crate_name)
    }

    #[inline]
    pub fn edition(self) -> Edition {
        HygieneData::with(|data| data.expn_data(data.outer_expn(self)).edition)
    }

    /// Returns whether this context originates in a foreign crate's external macro.
    ///
    /// This is used to test whether a lint should not even begin to figure out whether it should
    /// be reported on the current node.
    pub fn in_external_macro(self, sm: &SourceMap) -> bool {
        let expn_data = self.outer_expn_data();
        match expn_data.kind {
            ExpnKind::Root
            | ExpnKind::Desugaring(
                DesugaringKind::ForLoop
                | DesugaringKind::WhileLoop
                | DesugaringKind::OpaqueTy
                | DesugaringKind::Async
                | DesugaringKind::Await,
            ) => false,
            ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) => true, // well, it's "external"
            ExpnKind::Macro(MacroKind::Bang, _) => {
                // Dummy span for the `def_site` means it's an external macro.
                expn_data.def_site.is_dummy() || sm.is_imported(expn_data.def_site)
            }
            ExpnKind::Macro { .. } => true, // definitely a plugin
        }
    }
}

impl fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl Span {
    /// Reuses the span but adds information like the kind of the desugaring and features that are
    /// allowed inside this span.
    pub fn mark_with_reason(
        self,
        allow_internal_unstable: Option<Arc<[Symbol]>>,
        reason: DesugaringKind,
        edition: Edition,
        ctx: impl HashStableContext,
    ) -> Span {
        let expn_data = ExpnData {
            allow_internal_unstable,
            ..ExpnData::default(ExpnKind::Desugaring(reason), self, edition, None, None)
        };
        let expn_id = LocalExpnId::fresh(expn_data, ctx);
        self.apply_mark(expn_id.to_expn_id(), Transparency::Transparent)
    }
}

/// A subset of properties from both macro definition and macro call available through global data.
/// Avoid using this if you have access to the original definition or call structures.
#[derive(Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub struct ExpnData {
    // --- The part unique to each expansion.
    pub kind: ExpnKind,
    /// The expansion that contains the definition of the macro for this expansion.
    pub parent: ExpnId,
    /// The span of the macro call which produced this expansion.
    ///
    /// This span will typically have a different `ExpnData` and `call_site`.
    /// This recursively traces back through any macro calls which expanded into further
    /// macro calls, until the "source call-site" is reached at the root SyntaxContext.
    /// For example, if `food!()` expands to `fruit!()` which then expands to `grape`,
    /// then the call-site of `grape` is `fruit!()` and the call-site of `fruit!()`
    /// is `food!()`.
    ///
    /// For a desugaring expansion, this is the span of the expression or node that was
    /// desugared.
    pub call_site: Span,
    /// Used to force two `ExpnData`s to have different `Fingerprint`s.
    /// Due to macro expansion, it's possible to end up with two `ExpnId`s
    /// that have identical `ExpnData`s. This violates the contract of `HashStable`
    /// - the two `ExpnId`s are not equal, but their `Fingerprint`s are equal
    /// (since the numerical `ExpnId` value is not considered by the `HashStable`
    /// implementation).
    ///
    /// The `disambiguator` field is set by `update_disambiguator` when two distinct
    /// `ExpnId`s would end up with the same `Fingerprint`. Since `ExpnData` includes
    /// a `krate` field, this value only needs to be unique within a single crate.
    disambiguator: u32,

    // --- The part specific to the macro/desugaring definition.
    // --- It may be reasonable to share this part between expansions with the same definition,
    // --- but such sharing is known to bring some minor inconveniences without also bringing
    // --- noticeable perf improvements (PR #62898).
    /// The span of the macro definition (possibly dummy).
    /// This span serves only informational purpose and is not used for resolution.
    pub def_site: Span,
    /// List of `#[unstable]`/feature-gated features that the macro is allowed to use
    /// internally without forcing the whole crate to opt-in
    /// to them.
    pub allow_internal_unstable: Option<Arc<[Symbol]>>,
    /// Edition of the crate in which the macro is defined.
    pub edition: Edition,
    /// The `DefId` of the macro being invoked,
    /// if this `ExpnData` corresponds to a macro invocation
    pub macro_def_id: Option<DefId>,
    /// The normal module (`mod`) in which the expanded macro was defined.
    pub parent_module: Option<DefId>,
    /// Suppresses the `unsafe_code` lint for code produced by this macro.
    pub(crate) allow_internal_unsafe: bool,
    /// Enables the macro helper hack (`ident!(...)` -> `$crate::ident!(...)`) for this macro.
    pub local_inner_macros: bool,
    /// Should debuginfo for the macro be collapsed to the outermost expansion site (in other
    /// words, was the macro definition annotated with `#[collapse_debuginfo]`)?
    pub(crate) collapse_debuginfo: bool,
    /// When true, we do not display the note telling people to use the `-Zmacro-backtrace` flag.
    pub hide_backtrace: bool,
}

impl !PartialEq for ExpnData {}
impl !Hash for ExpnData {}

impl ExpnData {
    pub fn new(
        kind: ExpnKind,
        parent: ExpnId,
        call_site: Span,
        def_site: Span,
        allow_internal_unstable: Option<Arc<[Symbol]>>,
        edition: Edition,
        macro_def_id: Option<DefId>,
        parent_module: Option<DefId>,
        allow_internal_unsafe: bool,
        local_inner_macros: bool,
        collapse_debuginfo: bool,
        hide_backtrace: bool,
    ) -> ExpnData {
        ExpnData {
            kind,
            parent,
            call_site,
            def_site,
            allow_internal_unstable,
            edition,
            macro_def_id,
            parent_module,
            disambiguator: 0,
            allow_internal_unsafe,
            local_inner_macros,
            collapse_debuginfo,
            hide_backtrace,
        }
    }

    /// Constructs expansion data with default properties.
    pub fn default(
        kind: ExpnKind,
        call_site: Span,
        edition: Edition,
        macro_def_id: Option<DefId>,
        parent_module: Option<DefId>,
    ) -> ExpnData {
        ExpnData {
            kind,
            parent: ExpnId::root(),
            call_site,
            def_site: DUMMY_SP,
            allow_internal_unstable: None,
            edition,
            macro_def_id,
            parent_module,
            disambiguator: 0,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            collapse_debuginfo: false,
            hide_backtrace: false,
        }
    }

    pub fn allow_unstable(
        kind: ExpnKind,
        call_site: Span,
        edition: Edition,
        allow_internal_unstable: Arc<[Symbol]>,
        macro_def_id: Option<DefId>,
        parent_module: Option<DefId>,
    ) -> ExpnData {
        ExpnData {
            allow_internal_unstable: Some(allow_internal_unstable),
            ..ExpnData::default(kind, call_site, edition, macro_def_id, parent_module)
        }
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        matches!(self.kind, ExpnKind::Root)
    }

    #[inline]
    fn hash_expn(&self, ctx: &mut impl HashStableContext) -> Hash64 {
        let mut hasher = StableHasher::new();
        self.hash_stable(ctx, &mut hasher);
        hasher.finish()
    }
}

/// Expansion kind.
#[derive(Clone, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum ExpnKind {
    /// No expansion, aka root expansion. Only `ExpnId::root()` has this kind.
    Root,
    /// Expansion produced by a macro.
    Macro(MacroKind, Symbol),
    /// Transform done by the compiler on the AST.
    AstPass(AstPass),
    /// Desugaring done by the compiler during AST lowering.
    Desugaring(DesugaringKind),
}

impl ExpnKind {
    pub fn descr(&self) -> String {
        match *self {
            ExpnKind::Root => kw::PathRoot.to_string(),
            ExpnKind::Macro(macro_kind, name) => match macro_kind {
                MacroKind::Bang => format!("{name}!"),
                MacroKind::Attr => format!("#[{name}]"),
                MacroKind::Derive => format!("#[derive({name})]"),
            },
            ExpnKind::AstPass(kind) => kind.descr().to_string(),
            ExpnKind::Desugaring(kind) => format!("desugaring of {}", kind.descr()),
        }
    }
}

/// The kind of macro invocation or definition.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Encodable, Decodable, Hash, Debug)]
#[derive(HashStable_Generic)]
pub enum MacroKind {
    /// A bang macro `foo!()`.
    Bang,
    /// An attribute macro `#[foo]`.
    Attr,
    /// A derive macro `#[derive(Foo)]`
    Derive,
}

impl MacroKind {
    pub fn descr(self) -> &'static str {
        match self {
            MacroKind::Bang => "macro",
            MacroKind::Attr => "attribute macro",
            MacroKind::Derive => "derive macro",
        }
    }

    pub fn descr_expected(self) -> &'static str {
        match self {
            MacroKind::Attr => "attribute",
            _ => self.descr(),
        }
    }

    pub fn article(self) -> &'static str {
        match self {
            MacroKind::Attr => "an",
            _ => "a",
        }
    }
}

/// The kind of AST transform.
#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum AstPass {
    StdImports,
    TestHarness,
    ProcMacroHarness,
}

impl AstPass {
    pub fn descr(self) -> &'static str {
        match self {
            AstPass::StdImports => "standard library imports",
            AstPass::TestHarness => "test harness",
            AstPass::ProcMacroHarness => "proc macro harness",
        }
    }
}

/// The kind of compiler desugaring.
#[derive(Clone, Copy, PartialEq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum DesugaringKind {
    QuestionMark,
    TryBlock,
    YeetExpr,
    /// Desugaring of an `impl Trait` in return type position
    /// to an `type Foo = impl Trait;` and replacing the
    /// `impl Trait` with `Foo`.
    OpaqueTy,
    Async,
    Await,
    ForLoop,
    WhileLoop,
    /// `async Fn()` bound modifier
    BoundModifier,
    /// Calls to contract checks (`#[requires]` to precond, `#[ensures]` to postcond)
    Contract,
    /// A pattern type range start/end
    PatTyRange,
    /// A format literal.
    FormatLiteral {
        /// Was this format literal written in the source?
        /// - `format!("boo")` => Yes,
        /// - `format!(concat!("b", "o", "o"))` => No,
        /// - `format!(include_str!("boo.txt"))` => No,
        ///
        /// If it wasn't written in the source then we have to be careful with suggestions about
        /// rewriting it.
        source: bool,
    },
}

impl DesugaringKind {
    /// The description wording should combine well with "desugaring of {}".
    pub fn descr(self) -> &'static str {
        match self {
            DesugaringKind::Async => "`async` block or function",
            DesugaringKind::Await => "`await` expression",
            DesugaringKind::QuestionMark => "operator `?`",
            DesugaringKind::TryBlock => "`try` block",
            DesugaringKind::YeetExpr => "`do yeet` expression",
            DesugaringKind::OpaqueTy => "`impl Trait`",
            DesugaringKind::ForLoop => "`for` loop",
            DesugaringKind::WhileLoop => "`while` loop",
            DesugaringKind::BoundModifier => "trait bound modifier",
            DesugaringKind::Contract => "contract check",
            DesugaringKind::PatTyRange => "pattern type",
            DesugaringKind::FormatLiteral { source: true } => "format string literal",
            DesugaringKind::FormatLiteral { source: false } => {
                "expression that expanded into a format string literal"
            }
        }
    }

    /// For use with `rustc_unimplemented` to support conditions
    /// like `from_desugaring = "QuestionMark"`
    pub fn matches(&self, value: &str) -> bool {
        match self {
            DesugaringKind::Async => value == "Async",
            DesugaringKind::Await => value == "Await",
            DesugaringKind::QuestionMark => value == "QuestionMark",
            DesugaringKind::TryBlock => value == "TryBlock",
            DesugaringKind::YeetExpr => value == "YeetExpr",
            DesugaringKind::OpaqueTy => value == "OpaqueTy",
            DesugaringKind::ForLoop => value == "ForLoop",
            DesugaringKind::WhileLoop => value == "WhileLoop",
            DesugaringKind::BoundModifier => value == "BoundModifier",
            DesugaringKind::Contract => value == "Contract",
            DesugaringKind::PatTyRange => value == "PatTyRange",
            DesugaringKind::FormatLiteral { .. } => value == "FormatLiteral",
        }
    }
}

#[derive(Default)]
pub struct HygieneEncodeContext {
    /// All `SyntaxContexts` for which we have written `SyntaxContextData` into crate metadata.
    /// This is `None` after we finish encoding `SyntaxContexts`, to ensure
    /// that we don't accidentally try to encode any more `SyntaxContexts`
    serialized_ctxts: Lock<FxHashSet<SyntaxContext>>,
    /// The `SyntaxContexts` that we have serialized (e.g. as a result of encoding `Spans`)
    /// in the most recent 'round' of serializing. Serializing `SyntaxContextData`
    /// may cause us to serialize more `SyntaxContext`s, so serialize in a loop
    /// until we reach a fixed point.
    latest_ctxts: Lock<FxHashSet<SyntaxContext>>,

    serialized_expns: Lock<FxHashSet<ExpnId>>,

    latest_expns: Lock<FxHashSet<ExpnId>>,
}

impl HygieneEncodeContext {
    /// Record the fact that we need to serialize the corresponding `ExpnData`.
    pub fn schedule_expn_data_for_encoding(&self, expn: ExpnId) {
        if !self.serialized_expns.lock().contains(&expn) {
            self.latest_expns.lock().insert(expn);
        }
    }

    pub fn encode<T>(
        &self,
        encoder: &mut T,
        mut encode_ctxt: impl FnMut(&mut T, u32, &SyntaxContextKey),
        mut encode_expn: impl FnMut(&mut T, ExpnId, &ExpnData, ExpnHash),
    ) {
        // When we serialize a `SyntaxContextData`, we may end up serializing
        // a `SyntaxContext` that we haven't seen before
        while !self.latest_ctxts.lock().is_empty() || !self.latest_expns.lock().is_empty() {
            debug!(
                "encode_hygiene: Serializing a round of {:?} SyntaxContextData: {:?}",
                self.latest_ctxts.lock().len(),
                self.latest_ctxts
            );

            // Consume the current round of syntax contexts.
            // Drop the lock() temporary early.
            // It's fine to iterate over a HashMap, because the serialization of the table
            // that we insert data into doesn't depend on insertion order.
            #[allow(rustc::potential_query_instability)]
            let latest_ctxts = { mem::take(&mut *self.latest_ctxts.lock()) }.into_iter();
            let all_ctxt_data: Vec<_> = HygieneData::with(|data| {
                latest_ctxts
                    .map(|ctxt| (ctxt, data.syntax_context_data[ctxt.0 as usize].key()))
                    .collect()
            });
            for (ctxt, ctxt_key) in all_ctxt_data {
                if self.serialized_ctxts.lock().insert(ctxt) {
                    encode_ctxt(encoder, ctxt.0, &ctxt_key);
                }
            }

            // Same as above, but for expansions instead of syntax contexts.
            #[allow(rustc::potential_query_instability)]
            let latest_expns = { mem::take(&mut *self.latest_expns.lock()) }.into_iter();
            let all_expn_data: Vec<_> = HygieneData::with(|data| {
                latest_expns
                    .map(|expn| (expn, data.expn_data(expn).clone(), data.expn_hash(expn)))
                    .collect()
            });
            for (expn, expn_data, expn_hash) in all_expn_data {
                if self.serialized_expns.lock().insert(expn) {
                    encode_expn(encoder, expn, &expn_data, expn_hash);
                }
            }
        }
        debug!("encode_hygiene: Done serializing SyntaxContextData");
    }
}

/// Additional information used to assist in decoding hygiene data
#[derive(Default)]
pub struct HygieneDecodeContext {
    // A cache mapping raw serialized per-crate syntax context ids to corresponding decoded
    // `SyntaxContext`s in the current global `HygieneData`.
    remapped_ctxts: Lock<IndexVec<u32, Option<SyntaxContext>>>,
}

/// Register an expansion which has been decoded from the on-disk-cache for the local crate.
pub fn register_local_expn_id(data: ExpnData, hash: ExpnHash) -> ExpnId {
    HygieneData::with(|hygiene_data| {
        let expn_id = hygiene_data.local_expn_data.next_index();
        hygiene_data.local_expn_data.push(Some(data));
        let _eid = hygiene_data.local_expn_hashes.push(hash);
        debug_assert_eq!(expn_id, _eid);

        let expn_id = expn_id.to_expn_id();

        let _old_id = hygiene_data.expn_hash_to_expn_id.insert(hash, expn_id);
        debug_assert!(_old_id.is_none());
        expn_id
    })
}

/// Register an expansion which has been decoded from the metadata of a foreign crate.
pub fn register_expn_id(
    krate: CrateNum,
    local_id: ExpnIndex,
    data: ExpnData,
    hash: ExpnHash,
) -> ExpnId {
    debug_assert!(data.parent == ExpnId::root() || krate == data.parent.krate);
    let expn_id = ExpnId { krate, local_id };
    HygieneData::with(|hygiene_data| {
        let _old_data = hygiene_data.foreign_expn_data.insert(expn_id, data);
        let _old_hash = hygiene_data.foreign_expn_hashes.insert(expn_id, hash);
        debug_assert!(_old_hash.is_none() || _old_hash == Some(hash));
        let _old_id = hygiene_data.expn_hash_to_expn_id.insert(hash, expn_id);
        debug_assert!(_old_id.is_none() || _old_id == Some(expn_id));
    });
    expn_id
}

/// Decode an expansion from the metadata of a foreign crate.
pub fn decode_expn_id(
    krate: CrateNum,
    index: u32,
    decode_data: impl FnOnce(ExpnId) -> (ExpnData, ExpnHash),
) -> ExpnId {
    if index == 0 {
        trace!("decode_expn_id: deserialized root");
        return ExpnId::root();
    }

    let index = ExpnIndex::from_u32(index);

    // This function is used to decode metadata, so it cannot decode information about LOCAL_CRATE.
    debug_assert_ne!(krate, LOCAL_CRATE);
    let expn_id = ExpnId { krate, local_id: index };

    // Fast path if the expansion has already been decoded.
    if HygieneData::with(|hygiene_data| hygiene_data.foreign_expn_data.contains_key(&expn_id)) {
        return expn_id;
    }

    // Don't decode the data inside `HygieneData::with`, since we need to recursively decode
    // other ExpnIds
    let (expn_data, hash) = decode_data(expn_id);

    register_expn_id(krate, index, expn_data, hash)
}

// Decodes `SyntaxContext`, using the provided `HygieneDecodeContext`
// to track which `SyntaxContext`s we have already decoded.
// The provided closure will be invoked to deserialize a `SyntaxContextData`
// if we haven't already seen the id of the `SyntaxContext` we are deserializing.
pub fn decode_syntax_context<D: Decoder>(
    d: &mut D,
    context: &HygieneDecodeContext,
    decode_data: impl FnOnce(&mut D, u32) -> SyntaxContextKey,
) -> SyntaxContext {
    let raw_id: u32 = Decodable::decode(d);
    if raw_id == 0 {
        trace!("decode_syntax_context: deserialized root");
        // The root is special
        return SyntaxContext::root();
    }

    // Look into the cache first.
    // Reminder: `HygieneDecodeContext` is per-crate, so there are no collisions between
    // raw ids from different crate metadatas.
    if let Some(Some(ctxt)) = context.remapped_ctxts.lock().get(raw_id) {
        return *ctxt;
    }

    // Don't try to decode data while holding the lock, since we need to
    // be able to recursively decode a SyntaxContext
    let (parent, expn_id, transparency) = decode_data(d, raw_id);
    let ctxt =
        HygieneData::with(|hygiene_data| hygiene_data.alloc_ctxt(parent, expn_id, transparency));

    context.remapped_ctxts.lock().insert(raw_id, ctxt);

    ctxt
}

impl<E: SpanEncoder> Encodable<E> for LocalExpnId {
    fn encode(&self, e: &mut E) {
        self.to_expn_id().encode(e);
    }
}

impl<D: SpanDecoder> Decodable<D> for LocalExpnId {
    fn decode(d: &mut D) -> Self {
        ExpnId::expect_local(ExpnId::decode(d))
    }
}

pub fn raw_encode_syntax_context(
    ctxt: SyntaxContext,
    context: &HygieneEncodeContext,
    e: &mut impl Encoder,
) {
    if !context.serialized_ctxts.lock().contains(&ctxt) {
        context.latest_ctxts.lock().insert(ctxt);
    }
    ctxt.0.encode(e);
}

/// Updates the `disambiguator` field of the corresponding `ExpnData`
/// such that the `Fingerprint` of the `ExpnData` does not collide with
/// any other `ExpnIds`.
///
/// This method is called only when an `ExpnData` is first associated
/// with an `ExpnId` (when the `ExpnId` is initially constructed, or via
/// `set_expn_data`). It is *not* called for foreign `ExpnId`s deserialized
/// from another crate's metadata - since `ExpnHash` includes the stable crate id,
/// collisions are only possible between `ExpnId`s within the same crate.
fn update_disambiguator(expn_data: &mut ExpnData, mut ctx: impl HashStableContext) -> ExpnHash {
    // This disambiguator should not have been set yet.
    assert_eq!(expn_data.disambiguator, 0, "Already set disambiguator for ExpnData: {expn_data:?}");
    assert_default_hashing_controls(&ctx, "ExpnData (disambiguator)");
    let mut expn_hash = expn_data.hash_expn(&mut ctx);

    let disambiguator = HygieneData::with(|data| {
        // If this is the first ExpnData with a given hash, then keep our
        // disambiguator at 0 (the default u32 value)
        let disambig = data.expn_data_disambiguators.entry(expn_hash).or_default();
        let disambiguator = *disambig;
        *disambig += 1;
        disambiguator
    });

    if disambiguator != 0 {
        debug!("Set disambiguator for expn_data={:?} expn_hash={:?}", expn_data, expn_hash);

        expn_data.disambiguator = disambiguator;
        expn_hash = expn_data.hash_expn(&mut ctx);

        // Verify that the new disambiguator makes the hash unique
        #[cfg(debug_assertions)]
        HygieneData::with(|data| {
            assert_eq!(
                data.expn_data_disambiguators.get(&expn_hash),
                None,
                "Hash collision after disambiguator update!",
            );
        });
    }

    ExpnHash::new(ctx.def_path_hash(LOCAL_CRATE.as_def_id()).stable_crate_id(), expn_hash)
}

impl<CTX: HashStableContext> HashStable<CTX> for SyntaxContext {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        const TAG_EXPANSION: u8 = 0;
        const TAG_NO_EXPANSION: u8 = 1;

        if self.is_root() {
            TAG_NO_EXPANSION.hash_stable(ctx, hasher);
        } else {
            TAG_EXPANSION.hash_stable(ctx, hasher);
            let (expn_id, transparency) = self.outer_mark();
            expn_id.hash_stable(ctx, hasher);
            transparency.hash_stable(ctx, hasher);
        }
    }
}

impl<CTX: HashStableContext> HashStable<CTX> for ExpnId {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        assert_default_hashing_controls(ctx, "ExpnId");
        let hash = if *self == ExpnId::root() {
            // Avoid fetching TLS storage for a trivial often-used value.
            Fingerprint::ZERO
        } else {
            self.expn_hash().0
        };

        hash.hash_stable(ctx, hasher);
    }
}
