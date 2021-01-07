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

use crate::edition::Edition;
use crate::symbol::{kw, sym, Symbol};
use crate::SESSION_GLOBALS;
use crate::{Span, DUMMY_SP};

use crate::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::{Lock, Lrc};
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use std::fmt;
use tracing::*;

/// A `SyntaxContext` represents a chain of pairs `(ExpnId, Transparency)` named "marks".
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxContext(u32);

#[derive(Debug, Encodable, Decodable, Clone)]
pub struct SyntaxContextData {
    outer_expn: ExpnId,
    outer_transparency: Transparency,
    parent: SyntaxContext,
    /// This context, but with all transparent and semi-transparent expansions filtered away.
    opaque: SyntaxContext,
    /// This context, but with all transparent expansions filtered away.
    opaque_and_semitransparent: SyntaxContext,
    /// Name of the crate to which `$crate` with this context would resolve.
    dollar_crate_name: Symbol,
}

/// A unique ID associated with a macro invocation and expansion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpnId(u32);

/// A property of a macro expansion that determines how identifiers
/// produced by that expansion are resolved.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug, Encodable, Decodable)]
#[derive(HashStable_Generic)]
pub enum Transparency {
    /// Identifier produced by a transparent expansion is always resolved at call-site.
    /// Call-site spans in procedural macros, hygiene opt-out in `macro` should use this.
    Transparent,
    /// Identifier produced by a semi-transparent expansion may be resolved
    /// either at call-site or at definition-site.
    /// If it's a local variable, label or `$crate` then it's resolved at def-site.
    /// Otherwise it's resolved at call-site.
    /// `macro_rules` macros behave like this, built-in macros currently behave like this too,
    /// but that's an implementation detail.
    SemiTransparent,
    /// Identifier produced by an opaque expansion is always resolved at definition-site.
    /// Def-site spans in procedural macros, identifiers from `macro` by default use this.
    Opaque,
}

impl ExpnId {
    pub fn fresh(expn_data: Option<ExpnData>) -> Self {
        HygieneData::with(|data| data.fresh_expn(expn_data))
    }

    /// The ID of the theoretical expansion that generates freshly parsed, unexpanded AST.
    #[inline]
    pub fn root() -> Self {
        ExpnId(0)
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn from_u32(raw: u32) -> ExpnId {
        ExpnId(raw)
    }

    #[inline]
    pub fn expn_data(self) -> ExpnData {
        HygieneData::with(|data| data.expn_data(self).clone())
    }

    #[inline]
    pub fn set_expn_data(self, mut expn_data: ExpnData) {
        HygieneData::with(|data| {
            let old_expn_data = &mut data.expn_data[self.0 as usize];
            assert!(old_expn_data.is_none(), "expansion data is reset for an expansion ID");
            expn_data.orig_id.replace(self.as_u32()).expect_none("orig_id should be None");
            *old_expn_data = Some(expn_data);
        })
    }

    pub fn is_descendant_of(self, ancestor: ExpnId) -> bool {
        HygieneData::with(|data| data.is_descendant_of(self, ancestor))
    }

    /// `expn_id.outer_expn_is_descendant_of(ctxt)` is equivalent to but faster than
    /// `expn_id.is_descendant_of(ctxt.outer_expn())`.
    pub fn outer_expn_is_descendant_of(self, ctxt: SyntaxContext) -> bool {
        HygieneData::with(|data| data.is_descendant_of(self, data.outer_expn(ctxt)))
    }

    /// Returns span for the macro which originally caused this expansion to happen.
    ///
    /// Stops backtracing at include! boundary.
    pub fn expansion_cause(mut self) -> Option<Span> {
        let mut last_macro = None;
        loop {
            let expn_data = self.expn_data();
            // Stop going up the backtrace once include! is encountered
            if expn_data.is_root()
                || expn_data.kind == ExpnKind::Macro(MacroKind::Bang, sym::include)
            {
                break;
            }
            self = expn_data.call_site.ctxt().outer_expn();
            last_macro = Some(expn_data.call_site);
        }
        last_macro
    }
}

#[derive(Debug)]
pub struct HygieneData {
    /// Each expansion should have an associated expansion data, but sometimes there's a delay
    /// between creation of an expansion ID and obtaining its data (e.g. macros are collected
    /// first and then resolved later), so we use an `Option` here.
    expn_data: Vec<Option<ExpnData>>,
    syntax_context_data: Vec<SyntaxContextData>,
    syntax_context_map: FxHashMap<(SyntaxContext, ExpnId, Transparency), SyntaxContext>,
}

impl HygieneData {
    crate fn new(edition: Edition) -> Self {
        let mut root_data = ExpnData::default(
            ExpnKind::Root,
            DUMMY_SP,
            edition,
            Some(DefId::local(CRATE_DEF_INDEX)),
        );
        root_data.orig_id = Some(0);

        HygieneData {
            expn_data: vec![Some(root_data)],
            syntax_context_data: vec![SyntaxContextData {
                outer_expn: ExpnId::root(),
                outer_transparency: Transparency::Opaque,
                parent: SyntaxContext(0),
                opaque: SyntaxContext(0),
                opaque_and_semitransparent: SyntaxContext(0),
                dollar_crate_name: kw::DollarCrate,
            }],
            syntax_context_map: FxHashMap::default(),
        }
    }

    pub fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        SESSION_GLOBALS.with(|session_globals| f(&mut *session_globals.hygiene_data.borrow_mut()))
    }

    fn fresh_expn(&mut self, mut expn_data: Option<ExpnData>) -> ExpnId {
        let raw_id = self.expn_data.len() as u32;
        if let Some(data) = expn_data.as_mut() {
            data.orig_id.replace(raw_id).expect_none("orig_id should be None");
        }
        self.expn_data.push(expn_data);
        ExpnId(raw_id)
    }

    fn expn_data(&self, expn_id: ExpnId) -> &ExpnData {
        self.expn_data[expn_id.0 as usize].as_ref().expect("no expansion data for an expansion ID")
    }

    fn is_descendant_of(&self, mut expn_id: ExpnId, ancestor: ExpnId) -> bool {
        while expn_id != ancestor {
            if expn_id == ExpnId::root() {
                return false;
            }
            expn_id = self.expn_data(expn_id).parent;
        }
        true
    }

    fn normalize_to_macros_2_0(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].opaque
    }

    fn normalize_to_macro_rules(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].opaque_and_semitransparent
    }

    fn outer_expn(&self, ctxt: SyntaxContext) -> ExpnId {
        self.syntax_context_data[ctxt.0 as usize].outer_expn
    }

    fn outer_mark(&self, ctxt: SyntaxContext) -> (ExpnId, Transparency) {
        let data = &self.syntax_context_data[ctxt.0 as usize];
        (data.outer_expn, data.outer_transparency)
    }

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
        while ctxt != SyntaxContext::root() {
            debug!("marks: getting parent of {:?}", ctxt);
            marks.push(self.outer_mark(ctxt));
            ctxt = self.parent_ctxt(ctxt);
        }
        marks.reverse();
        marks
    }

    fn walk_chain(&self, mut span: Span, to: SyntaxContext) -> Span {
        debug!("walk_chain({:?}, {:?})", span, to);
        debug!("walk_chain: span ctxt = {:?}", span.ctxt());
        while span.from_expansion() && span.ctxt() != to {
            let outer_expn = self.outer_expn(span.ctxt());
            debug!("walk_chain({:?}): outer_expn={:?}", span, outer_expn);
            let expn_data = self.expn_data(outer_expn);
            debug!("walk_chain({:?}): expn_data={:?}", span, expn_data);
            span = expn_data.call_site;
        }
        span
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
            return self.apply_mark_internal(ctxt, expn_id, transparency);
        }

        let call_site_ctxt = self.expn_data(expn_id).call_site.ctxt();
        let mut call_site_ctxt = if transparency == Transparency::SemiTransparent {
            self.normalize_to_macros_2_0(call_site_ctxt)
        } else {
            self.normalize_to_macro_rules(call_site_ctxt)
        };

        if call_site_ctxt == SyntaxContext::root() {
            return self.apply_mark_internal(ctxt, expn_id, transparency);
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
            call_site_ctxt = self.apply_mark_internal(call_site_ctxt, expn_id, transparency);
        }
        self.apply_mark_internal(call_site_ctxt, expn_id, transparency)
    }

    fn apply_mark_internal(
        &mut self,
        ctxt: SyntaxContext,
        expn_id: ExpnId,
        transparency: Transparency,
    ) -> SyntaxContext {
        let syntax_context_data = &mut self.syntax_context_data;
        let mut opaque = syntax_context_data[ctxt.0 as usize].opaque;
        let mut opaque_and_semitransparent =
            syntax_context_data[ctxt.0 as usize].opaque_and_semitransparent;

        if transparency >= Transparency::Opaque {
            let parent = opaque;
            opaque = *self
                .syntax_context_map
                .entry((parent, expn_id, transparency))
                .or_insert_with(|| {
                    let new_opaque = SyntaxContext(syntax_context_data.len() as u32);
                    syntax_context_data.push(SyntaxContextData {
                        outer_expn: expn_id,
                        outer_transparency: transparency,
                        parent,
                        opaque: new_opaque,
                        opaque_and_semitransparent: new_opaque,
                        dollar_crate_name: kw::DollarCrate,
                    });
                    new_opaque
                });
        }

        if transparency >= Transparency::SemiTransparent {
            let parent = opaque_and_semitransparent;
            opaque_and_semitransparent = *self
                .syntax_context_map
                .entry((parent, expn_id, transparency))
                .or_insert_with(|| {
                    let new_opaque_and_semitransparent =
                        SyntaxContext(syntax_context_data.len() as u32);
                    syntax_context_data.push(SyntaxContextData {
                        outer_expn: expn_id,
                        outer_transparency: transparency,
                        parent,
                        opaque,
                        opaque_and_semitransparent: new_opaque_and_semitransparent,
                        dollar_crate_name: kw::DollarCrate,
                    });
                    new_opaque_and_semitransparent
                });
        }

        let parent = ctxt;
        *self.syntax_context_map.entry((parent, expn_id, transparency)).or_insert_with(|| {
            let new_opaque_and_semitransparent_and_transparent =
                SyntaxContext(syntax_context_data.len() as u32);
            syntax_context_data.push(SyntaxContextData {
                outer_expn: expn_id,
                outer_transparency: transparency,
                parent,
                opaque,
                opaque_and_semitransparent,
                dollar_crate_name: kw::DollarCrate,
            });
            new_opaque_and_semitransparent_and_transparent
        })
    }
}

pub fn clear_syntax_context_map() {
    HygieneData::with(|data| data.syntax_context_map = FxHashMap::default());
}

pub fn walk_chain(span: Span, to: SyntaxContext) -> Span {
    HygieneData::with(|data| data.walk_chain(span, to))
}

pub fn update_dollar_crate_names(mut get_name: impl FnMut(SyntaxContext) -> Symbol) {
    // The new contexts that need updating are at the end of the list and have `$crate` as a name.
    let (len, to_update) = HygieneData::with(|data| {
        (
            data.syntax_context_data.len(),
            data.syntax_context_data
                .iter()
                .rev()
                .take_while(|scdata| scdata.dollar_crate_name == kw::DollarCrate)
                .count(),
        )
    });
    // The callback must be called from outside of the `HygieneData` lock,
    // since it will try to acquire it too.
    let range_to_update = len - to_update..len;
    let names: Vec<_> =
        range_to_update.clone().map(|idx| get_name(SyntaxContext::from_u32(idx as u32))).collect();
    HygieneData::with(|data| {
        range_to_update.zip(names.into_iter()).for_each(|(idx, name)| {
            data.syntax_context_data[idx].dollar_crate_name = name;
        })
    })
}

pub fn debug_hygiene_data(verbose: bool) -> String {
    HygieneData::with(|data| {
        if verbose {
            format!("{:#?}", data)
        } else {
            let mut s = String::from("");
            s.push_str("Expansions:");
            data.expn_data.iter().enumerate().for_each(|(id, expn_info)| {
                let expn_info = expn_info.as_ref().expect("no expansion data for an expansion ID");
                s.push_str(&format!(
                    "\n{}: parent: {:?}, call_site_ctxt: {:?}, def_site_ctxt: {:?}, kind: {:?}",
                    id,
                    expn_info.parent,
                    expn_info.call_site.ctxt(),
                    expn_info.def_site.ctxt(),
                    expn_info.kind,
                ));
            });
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
    crate fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    crate fn from_u32(raw: u32) -> SyntaxContext {
        SyntaxContext(raw)
    }

    /// Extend a syntax context with a given expansion and transparency.
    crate fn apply_mark(self, expn_id: ExpnId, transparency: Transparency) -> SyntaxContext {
        HygieneData::with(|data| data.apply_mark(self, expn_id, transparency))
    }

    /// Pulls a single mark off of the syntax context. This effectively moves the
    /// context up one macro definition level. That is, if we have a nested macro
    /// definition as follows:
    ///
    /// ```rust
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
    pub fn remove_mark(&mut self) -> ExpnId {
        HygieneData::with(|data| data.remove_mark(self).0)
    }

    pub fn marks(self) -> Vec<(ExpnId, Transparency)> {
        HygieneData::with(|data| data.marks(self))
    }

    /// Adjust this context for resolution in a scope created by the given expansion.
    /// For example, consider the following three resolutions of `f`:
    ///
    /// ```rust
    /// mod foo { pub fn f() {} } // `f`'s `SyntaxContext` is empty.
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
    pub fn adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        HygieneData::with(|data| data.adjust(self, expn_id))
    }

    /// Like `SyntaxContext::adjust`, but also normalizes `self` to macros 2.0.
    pub fn normalize_to_macros_2_0_and_adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        HygieneData::with(|data| {
            *self = data.normalize_to_macros_2_0(*self);
            data.adjust(self, expn_id)
        })
    }

    /// Adjust this context for resolution in a scope created by the given expansion
    /// via a glob import with the given `SyntaxContext`.
    /// For example:
    ///
    /// ```rust
    /// m!(f);
    /// macro m($i:ident) {
    ///     mod foo {
    ///         pub fn f() {} // `f`'s `SyntaxContext` has a single `ExpnId` from `m`.
    ///         pub fn $i() {} // `$i`'s `SyntaxContext` is empty.
    ///     }
    ///     n(f);
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
    pub fn glob_adjust(&mut self, expn_id: ExpnId, glob_span: Span) -> Option<Option<ExpnId>> {
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
    /// ```rust
    /// if let Some(privacy_checking_scope) = self.reverse_glob_adjust(expansion, glob_ctxt) {
    ///     assert!(self.glob_adjust(expansion, glob_ctxt) == Some(privacy_checking_scope));
    /// }
    /// ```
    pub fn reverse_glob_adjust(
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

    #[inline]
    pub fn outer_mark(self) -> (ExpnId, Transparency) {
        HygieneData::with(|data| data.outer_mark(self))
    }

    pub fn dollar_crate_name(self) -> Symbol {
        HygieneData::with(|data| data.syntax_context_data[self.0 as usize].dollar_crate_name)
    }
}

impl fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl Span {
    /// Creates a fresh expansion with given properties.
    /// Expansions are normally created by macros, but in some cases expansions are created for
    /// other compiler-generated code to set per-span properties like allowed unstable features.
    /// The returned span belongs to the created expansion and has the new properties,
    /// but its location is inherited from the current span.
    pub fn fresh_expansion(self, expn_data: ExpnData) -> Span {
        self.fresh_expansion_with_transparency(expn_data, Transparency::Transparent)
    }

    pub fn fresh_expansion_with_transparency(
        self,
        expn_data: ExpnData,
        transparency: Transparency,
    ) -> Span {
        HygieneData::with(|data| {
            let expn_id = data.fresh_expn(Some(expn_data));
            self.with_ctxt(data.apply_mark(SyntaxContext::root(), expn_id, transparency))
        })
    }

    /// Reuses the span but adds information like the kind of the desugaring and features that are
    /// allowed inside this span.
    pub fn mark_with_reason(
        self,
        allow_internal_unstable: Option<Lrc<[Symbol]>>,
        reason: DesugaringKind,
        edition: Edition,
    ) -> Span {
        self.fresh_expansion(ExpnData {
            allow_internal_unstable,
            ..ExpnData::default(ExpnKind::Desugaring(reason), self, edition, None)
        })
    }
}

/// A subset of properties from both macro definition and macro call available through global data.
/// Avoid using this if you have access to the original definition or call structures.
#[derive(Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub struct ExpnData {
    // --- The part unique to each expansion.
    /// The kind of this expansion - macro or compiler desugaring.
    pub kind: ExpnKind,
    /// The expansion that produced this expansion.
    pub parent: ExpnId,
    /// The location of the actual macro invocation or syntax sugar , e.g.
    /// `let x = foo!();` or `if let Some(y) = x {}`
    ///
    /// This may recursively refer to other macro invocations, e.g., if
    /// `foo!()` invoked `bar!()` internally, and there was an
    /// expression inside `bar!`; the call_site of the expression in
    /// the expansion would point to the `bar!` invocation; that
    /// call_site span would have its own ExpnData, with the call_site
    /// pointing to the `foo!` invocation.
    pub call_site: Span,

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
    pub allow_internal_unstable: Option<Lrc<[Symbol]>>,
    /// Whether the macro is allowed to use `unsafe` internally
    /// even if the user crate has `#![forbid(unsafe_code)]`.
    pub allow_internal_unsafe: bool,
    /// Enables the macro helper hack (`ident!(...)` -> `$crate::ident!(...)`)
    /// for a given macro.
    pub local_inner_macros: bool,
    /// Edition of the crate in which the macro is defined.
    pub edition: Edition,
    /// The `DefId` of the macro being invoked,
    /// if this `ExpnData` corresponds to a macro invocation
    pub macro_def_id: Option<DefId>,
    /// The crate that originally created this `ExpnData`. During
    /// metadata serialization, we only encode `ExpnData`s that were
    /// created locally - when our serialized metadata is decoded,
    /// foreign `ExpnId`s will have their `ExpnData` looked up
    /// from the crate specified by `Crate
    krate: CrateNum,
    /// The raw that this `ExpnData` had in its original crate.
    /// An `ExpnData` can be created before being assigned an `ExpnId`,
    /// so this might be `None` until `set_expn_data` is called
    // This is used only for serialization/deserialization purposes:
    // two `ExpnData`s that differ only in their `orig_id` should
    // be considered equivalent.
    #[stable_hasher(ignore)]
    orig_id: Option<u32>,
}

// This would require special handling of `orig_id` and `parent`
impl !PartialEq for ExpnData {}

impl ExpnData {
    pub fn new(
        kind: ExpnKind,
        parent: ExpnId,
        call_site: Span,
        def_site: Span,
        allow_internal_unstable: Option<Lrc<[Symbol]>>,
        allow_internal_unsafe: bool,
        local_inner_macros: bool,
        edition: Edition,
        macro_def_id: Option<DefId>,
    ) -> ExpnData {
        ExpnData {
            kind,
            parent,
            call_site,
            def_site,
            allow_internal_unstable,
            allow_internal_unsafe,
            local_inner_macros,
            edition,
            macro_def_id,
            krate: LOCAL_CRATE,
            orig_id: None,
        }
    }

    /// Constructs expansion data with default properties.
    pub fn default(
        kind: ExpnKind,
        call_site: Span,
        edition: Edition,
        macro_def_id: Option<DefId>,
    ) -> ExpnData {
        ExpnData {
            kind,
            parent: ExpnId::root(),
            call_site,
            def_site: DUMMY_SP,
            allow_internal_unstable: None,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            edition,
            macro_def_id,
            krate: LOCAL_CRATE,
            orig_id: None,
        }
    }

    pub fn allow_unstable(
        kind: ExpnKind,
        call_site: Span,
        edition: Edition,
        allow_internal_unstable: Lrc<[Symbol]>,
        macro_def_id: Option<DefId>,
    ) -> ExpnData {
        ExpnData {
            allow_internal_unstable: Some(allow_internal_unstable),
            ..ExpnData::default(kind, call_site, edition, macro_def_id)
        }
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        matches!(self.kind, ExpnKind::Root)
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
    /// Desugaring done by the compiler during HIR lowering.
    Desugaring(DesugaringKind),
    /// MIR inlining
    Inlined,
}

impl ExpnKind {
    pub fn descr(&self) -> String {
        match *self {
            ExpnKind::Root => kw::PathRoot.to_string(),
            ExpnKind::Macro(macro_kind, name) => match macro_kind {
                MacroKind::Bang => format!("{}!", name),
                MacroKind::Attr => format!("#[{}]", name),
                MacroKind::Derive => format!("#[derive({})]", name),
            },
            ExpnKind::AstPass(kind) => kind.descr().to_string(),
            ExpnKind::Desugaring(kind) => format!("desugaring of {}", kind.descr()),
            ExpnKind::Inlined => "inlined source".to_string(),
        }
    }
}

/// The kind of macro invocation or definition.
#[derive(Clone, Copy, PartialEq, Eq, Encodable, Decodable, Hash, Debug)]
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
    fn descr(self) -> &'static str {
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
    /// We desugar `if c { i } else { e }` to `match $ExprKind::Use(c) { true => i, _ => e }`.
    /// However, we do not want to blame `c` for unreachability but rather say that `i`
    /// is unreachable. This desugaring kind allows us to avoid blaming `c`.
    /// This also applies to `while` loops.
    CondTemporary,
    QuestionMark,
    TryBlock,
    /// Desugaring of an `impl Trait` in return type position
    /// to an `type Foo = impl Trait;` and replacing the
    /// `impl Trait` with `Foo`.
    OpaqueTy,
    Async,
    Await,
    ForLoop(ForLoopLoc),
}

/// A location in the desugaring of a `for` loop
#[derive(Clone, Copy, PartialEq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum ForLoopLoc {
    Head,
    IntoIter,
}

impl DesugaringKind {
    /// The description wording should combine well with "desugaring of {}".
    fn descr(self) -> &'static str {
        match self {
            DesugaringKind::CondTemporary => "`if` or `while` condition",
            DesugaringKind::Async => "`async` block or function",
            DesugaringKind::Await => "`await` expression",
            DesugaringKind::QuestionMark => "operator `?`",
            DesugaringKind::TryBlock => "`try` block",
            DesugaringKind::OpaqueTy => "`impl Trait`",
            DesugaringKind::ForLoop(_) => "`for` loop",
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
    /// in the most recent 'round' of serializnig. Serializing `SyntaxContextData`
    /// may cause us to serialize more `SyntaxContext`s, so serialize in a loop
    /// until we reach a fixed point.
    latest_ctxts: Lock<FxHashSet<SyntaxContext>>,

    serialized_expns: Lock<FxHashSet<ExpnId>>,

    latest_expns: Lock<FxHashSet<ExpnId>>,
}

impl HygieneEncodeContext {
    pub fn encode<
        T,
        R,
        F: FnMut(&mut T, u32, &SyntaxContextData) -> Result<(), R>,
        G: FnMut(&mut T, u32, &ExpnData) -> Result<(), R>,
    >(
        &self,
        encoder: &mut T,
        mut encode_ctxt: F,
        mut encode_expn: G,
    ) -> Result<(), R> {
        // When we serialize a `SyntaxContextData`, we may end up serializing
        // a `SyntaxContext` that we haven't seen before
        while !self.latest_ctxts.lock().is_empty() || !self.latest_expns.lock().is_empty() {
            debug!(
                "encode_hygiene: Serializing a round of {:?} SyntaxContextDatas: {:?}",
                self.latest_ctxts.lock().len(),
                self.latest_ctxts
            );

            // Consume the current round of SyntaxContexts.
            // Drop the lock() temporary early
            let latest_ctxts = { std::mem::take(&mut *self.latest_ctxts.lock()) };

            // It's fine to iterate over a HashMap, because the serialization
            // of the table that we insert data into doesn't depend on insertion
            // order
            for_all_ctxts_in(latest_ctxts.into_iter(), |(index, ctxt, data)| {
                if self.serialized_ctxts.lock().insert(ctxt) {
                    encode_ctxt(encoder, index, data)?;
                }
                Ok(())
            })?;

            let latest_expns = { std::mem::take(&mut *self.latest_expns.lock()) };

            for_all_expns_in(latest_expns.into_iter(), |index, expn, data| {
                if self.serialized_expns.lock().insert(expn) {
                    encode_expn(encoder, index, data)?;
                }
                Ok(())
            })?;
        }
        debug!("encode_hygiene: Done serializing SyntaxContextData");
        Ok(())
    }
}

#[derive(Default)]
/// Additional information used to assist in decoding hygiene data
pub struct HygieneDecodeContext {
    // Maps serialized `SyntaxContext` ids to a `SyntaxContext` in the current
    // global `HygieneData`. When we deserialize a `SyntaxContext`, we need to create
    // a new id in the global `HygieneData`. This map tracks the ID we end up picking,
    // so that multiple occurrences of the same serialized id are decoded to the same
    // `SyntaxContext`
    remapped_ctxts: Lock<Vec<Option<SyntaxContext>>>,
    // The same as `remapepd_ctxts`, but for `ExpnId`s
    remapped_expns: Lock<Vec<Option<ExpnId>>>,
}

pub fn decode_expn_id<
    'a,
    D: Decoder,
    F: FnOnce(&mut D, u32) -> Result<ExpnData, D::Error>,
    G: FnOnce(CrateNum) -> &'a HygieneDecodeContext,
>(
    d: &mut D,
    mode: ExpnDataDecodeMode<'a, G>,
    decode_data: F,
) -> Result<ExpnId, D::Error> {
    let index = u32::decode(d)?;
    let context = match mode {
        ExpnDataDecodeMode::IncrComp(context) => context,
        ExpnDataDecodeMode::Metadata(get_context) => {
            let krate = CrateNum::decode(d)?;
            get_context(krate)
        }
    };

    // Do this after decoding, so that we decode a `CrateNum`
    // if necessary
    if index == ExpnId::root().as_u32() {
        debug!("decode_expn_id: deserialized root");
        return Ok(ExpnId::root());
    }

    let outer_expns = &context.remapped_expns;

    // Ensure that the lock() temporary is dropped early
    {
        if let Some(expn_id) = outer_expns.lock().get(index as usize).copied().flatten() {
            return Ok(expn_id);
        }
    }

    // Don't decode the data inside `HygieneData::with`, since we need to recursively decode
    // other ExpnIds
    let mut expn_data = decode_data(d, index)?;

    let expn_id = HygieneData::with(|hygiene_data| {
        let expn_id = ExpnId(hygiene_data.expn_data.len() as u32);

        // If we just deserialized an `ExpnData` owned by
        // the local crate, its `orig_id` will be stale,
        // so we need to update it to its own value.
        // This only happens when we deserialize the incremental cache,
        // since a crate will never decode its own metadata.
        if expn_data.krate == LOCAL_CRATE {
            expn_data.orig_id = Some(expn_id.0);
        }

        hygiene_data.expn_data.push(Some(expn_data));

        let mut expns = outer_expns.lock();
        let new_len = index as usize + 1;
        if expns.len() < new_len {
            expns.resize(new_len, None);
        }
        expns[index as usize] = Some(expn_id);
        drop(expns);
        expn_id
    });
    Ok(expn_id)
}

// Decodes `SyntaxContext`, using the provided `HygieneDecodeContext`
// to track which `SyntaxContext`s we have already decoded.
// The provided closure will be invoked to deserialize a `SyntaxContextData`
// if we haven't already seen the id of the `SyntaxContext` we are deserializing.
pub fn decode_syntax_context<
    D: Decoder,
    F: FnOnce(&mut D, u32) -> Result<SyntaxContextData, D::Error>,
>(
    d: &mut D,
    context: &HygieneDecodeContext,
    decode_data: F,
) -> Result<SyntaxContext, D::Error> {
    let raw_id: u32 = Decodable::decode(d)?;
    if raw_id == 0 {
        debug!("decode_syntax_context: deserialized root");
        // The root is special
        return Ok(SyntaxContext::root());
    }

    let outer_ctxts = &context.remapped_ctxts;

    // Ensure that the lock() temporary is dropped early
    {
        if let Some(ctxt) = outer_ctxts.lock().get(raw_id as usize).copied().flatten() {
            return Ok(ctxt);
        }
    }

    // Allocate and store SyntaxContext id *before* calling the decoder function,
    // as the SyntaxContextData may reference itself.
    let new_ctxt = HygieneData::with(|hygiene_data| {
        let new_ctxt = SyntaxContext(hygiene_data.syntax_context_data.len() as u32);
        // Push a dummy SyntaxContextData to ensure that nobody else can get the
        // same ID as us. This will be overwritten after call `decode_Data`
        hygiene_data.syntax_context_data.push(SyntaxContextData {
            outer_expn: ExpnId::root(),
            outer_transparency: Transparency::Transparent,
            parent: SyntaxContext::root(),
            opaque: SyntaxContext::root(),
            opaque_and_semitransparent: SyntaxContext::root(),
            dollar_crate_name: kw::Empty,
        });
        let mut ctxts = outer_ctxts.lock();
        let new_len = raw_id as usize + 1;
        if ctxts.len() < new_len {
            ctxts.resize(new_len, None);
        }
        ctxts[raw_id as usize] = Some(new_ctxt);
        drop(ctxts);
        new_ctxt
    });

    // Don't try to decode data while holding the lock, since we need to
    // be able to recursively decode a SyntaxContext
    let mut ctxt_data = decode_data(d, raw_id)?;
    // Reset `dollar_crate_name` so that it will be updated by `update_dollar_crate_names`
    // We don't care what the encoding crate set this to - we want to resolve it
    // from the perspective of the current compilation session
    ctxt_data.dollar_crate_name = kw::DollarCrate;

    // Overwrite the dummy data with our decoded SyntaxContextData
    HygieneData::with(|hygiene_data| {
        let dummy = std::mem::replace(
            &mut hygiene_data.syntax_context_data[new_ctxt.as_u32() as usize],
            ctxt_data,
        );
        // Make sure nothing weird happening while `decode_data` was running
        assert_eq!(dummy.dollar_crate_name, kw::Empty);
    });

    Ok(new_ctxt)
}

pub fn num_syntax_ctxts() -> usize {
    HygieneData::with(|data| data.syntax_context_data.len())
}

pub fn for_all_ctxts_in<E, F: FnMut((u32, SyntaxContext, &SyntaxContextData)) -> Result<(), E>>(
    ctxts: impl Iterator<Item = SyntaxContext>,
    mut f: F,
) -> Result<(), E> {
    let all_data: Vec<_> = HygieneData::with(|data| {
        ctxts.map(|ctxt| (ctxt, data.syntax_context_data[ctxt.0 as usize].clone())).collect()
    });
    for (ctxt, data) in all_data.into_iter() {
        f((ctxt.0, ctxt, &data))?;
    }
    Ok(())
}

pub fn for_all_expns_in<E, F: FnMut(u32, ExpnId, &ExpnData) -> Result<(), E>>(
    expns: impl Iterator<Item = ExpnId>,
    mut f: F,
) -> Result<(), E> {
    let all_data: Vec<_> = HygieneData::with(|data| {
        expns.map(|expn| (expn, data.expn_data[expn.0 as usize].clone())).collect()
    });
    for (expn, data) in all_data.into_iter() {
        f(expn.0, expn, &data.unwrap_or_else(|| panic!("Missing data for {:?}", expn)))?;
    }
    Ok(())
}

pub fn for_all_data<E, F: FnMut((u32, SyntaxContext, &SyntaxContextData)) -> Result<(), E>>(
    mut f: F,
) -> Result<(), E> {
    let all_data = HygieneData::with(|data| data.syntax_context_data.clone());
    for (i, data) in all_data.into_iter().enumerate() {
        f((i as u32, SyntaxContext(i as u32), &data))?;
    }
    Ok(())
}

impl<E: Encoder> Encodable<E> for ExpnId {
    default fn encode(&self, _: &mut E) -> Result<(), E::Error> {
        panic!("cannot encode `ExpnId` with `{}`", std::any::type_name::<E>());
    }
}

impl<D: Decoder> Decodable<D> for ExpnId {
    default fn decode(_: &mut D) -> Result<Self, D::Error> {
        panic!("cannot decode `ExpnId` with `{}`", std::any::type_name::<D>());
    }
}

pub fn for_all_expn_data<E, F: FnMut(u32, &ExpnData) -> Result<(), E>>(mut f: F) -> Result<(), E> {
    let all_data = HygieneData::with(|data| data.expn_data.clone());
    for (i, data) in all_data.into_iter().enumerate() {
        f(i as u32, &data.unwrap_or_else(|| panic!("Missing ExpnData!")))?;
    }
    Ok(())
}

pub fn raw_encode_syntax_context<E: Encoder>(
    ctxt: SyntaxContext,
    context: &HygieneEncodeContext,
    e: &mut E,
) -> Result<(), E::Error> {
    if !context.serialized_ctxts.lock().contains(&ctxt) {
        context.latest_ctxts.lock().insert(ctxt);
    }
    ctxt.0.encode(e)
}

pub fn raw_encode_expn_id<E: Encoder>(
    expn: ExpnId,
    context: &HygieneEncodeContext,
    mode: ExpnDataEncodeMode,
    e: &mut E,
) -> Result<(), E::Error> {
    // Record the fact that we need to serialize the corresponding
    // `ExpnData`
    let needs_data = || {
        if !context.serialized_expns.lock().contains(&expn) {
            context.latest_expns.lock().insert(expn);
        }
    };

    match mode {
        ExpnDataEncodeMode::IncrComp => {
            // Always serialize the `ExpnData` in incr comp mode
            needs_data();
            expn.0.encode(e)
        }
        ExpnDataEncodeMode::Metadata => {
            let data = expn.expn_data();
            // We only need to serialize the ExpnData
            // if it comes from this crate.
            // We currently don't serialize any hygiene information data for
            // proc-macro crates: see the `SpecializedEncoder<Span>` impl
            // for crate metadata.
            if data.krate == LOCAL_CRATE {
                needs_data();
            }
            data.orig_id.expect("Missing orig_id").encode(e)?;
            data.krate.encode(e)
        }
    }
}

pub enum ExpnDataEncodeMode {
    IncrComp,
    Metadata,
}

pub enum ExpnDataDecodeMode<'a, F: FnOnce(CrateNum) -> &'a HygieneDecodeContext> {
    IncrComp(&'a HygieneDecodeContext),
    Metadata(F),
}

impl<'a> ExpnDataDecodeMode<'a, Box<dyn FnOnce(CrateNum) -> &'a HygieneDecodeContext>> {
    pub fn incr_comp(ctxt: &'a HygieneDecodeContext) -> Self {
        ExpnDataDecodeMode::IncrComp(ctxt)
    }
}

impl<E: Encoder> Encodable<E> for SyntaxContext {
    default fn encode(&self, _: &mut E) -> Result<(), E::Error> {
        panic!("cannot encode `SyntaxContext` with `{}`", std::any::type_name::<E>());
    }
}

impl<D: Decoder> Decodable<D> for SyntaxContext {
    default fn decode(_: &mut D) -> Result<Self, D::Error> {
        panic!("cannot decode `SyntaxContext` with `{}`", std::any::type_name::<D>());
    }
}
