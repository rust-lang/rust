//! Machinery for hygienic macros, inspired by the `MTWT[1]` paper.
//!
//! `[1]` Matthew Flatt, Ryan Culpepper, David Darais, and Robert Bruce Findler. 2012.
//! *Macros that work together: Compile-time bindings, partial expansion,
//! and definition contexts*. J. Funct. Program. 22, 2 (March 2012), 181-216.
//! DOI=10.1017/S0956796812000093 <https://doi.org/10.1017/S0956796812000093>

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

use crate::GLOBALS;
use crate::{Span, DUMMY_SP};
use crate::edition::Edition;
use crate::symbol::{kw, Symbol};

use rustc_serialize::{Encodable, Decodable, Encoder, Decoder};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use std::fmt;

/// A `SyntaxContext` represents a chain of pairs `(ExpnId, Transparency)` named "marks".
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxContext(u32);

#[derive(Debug)]
struct SyntaxContextData {
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
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Hash, Debug, RustcEncodable, RustcDecodable)]
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
    pub fn set_expn_data(self, expn_data: ExpnData) {
        HygieneData::with(|data| {
            let old_expn_data = &mut data.expn_data[self.0 as usize];
            assert!(old_expn_data.is_none(), "expansion data is reset for an expansion ID");
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

    // Used for enabling some compatibility fallback in resolve.
    #[inline]
    pub fn looks_like_proc_macro_derive(self) -> bool {
        HygieneData::with(|data| {
            let expn_data = data.expn_data(self);
            if let ExpnKind::Macro(MacroKind::Derive, _) = expn_data.kind {
                return expn_data.default_transparency == Transparency::Opaque;
            }
            false
        })
    }
}

#[derive(Debug)]
crate struct HygieneData {
    /// Each expansion should have an associated expansion data, but sometimes there's a delay
    /// between creation of an expansion ID and obtaining its data (e.g. macros are collected
    /// first and then resolved later), so we use an `Option` here.
    expn_data: Vec<Option<ExpnData>>,
    syntax_context_data: Vec<SyntaxContextData>,
    syntax_context_map: FxHashMap<(SyntaxContext, ExpnId, Transparency), SyntaxContext>,
}

impl HygieneData {
    crate fn new(edition: Edition) -> Self {
        HygieneData {
            expn_data: vec![Some(ExpnData::default(ExpnKind::Root, DUMMY_SP, edition))],
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

    fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        GLOBALS.with(|globals| f(&mut *globals.hygiene_data.borrow_mut()))
    }

    fn fresh_expn(&mut self, expn_data: Option<ExpnData>) -> ExpnId {
        self.expn_data.push(expn_data);
        ExpnId(self.expn_data.len() as u32 - 1)
    }

    fn expn_data(&self, expn_id: ExpnId) -> &ExpnData {
        self.expn_data[expn_id.0 as usize].as_ref()
            .expect("no expansion data for an expansion ID")
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

    fn modern(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].opaque
    }

    fn modern_and_legacy(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].opaque_and_semitransparent
    }

    fn outer_expn(&self, ctxt: SyntaxContext) -> ExpnId {
        self.syntax_context_data[ctxt.0 as usize].outer_expn
    }

    fn outer_transparency(&self, ctxt: SyntaxContext) -> Transparency {
        self.syntax_context_data[ctxt.0 as usize].outer_transparency
    }

    fn parent_ctxt(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_context_data[ctxt.0 as usize].parent
    }

    fn remove_mark(&self, ctxt: &mut SyntaxContext) -> ExpnId {
        let outer_expn = self.outer_expn(*ctxt);
        *ctxt = self.parent_ctxt(*ctxt);
        outer_expn
    }

    fn marks(&self, mut ctxt: SyntaxContext) -> Vec<(ExpnId, Transparency)> {
        let mut marks = Vec::new();
        while ctxt != SyntaxContext::root() {
            marks.push((self.outer_expn(ctxt), self.outer_transparency(ctxt)));
            ctxt = self.parent_ctxt(ctxt);
        }
        marks.reverse();
        marks
    }

    fn walk_chain(&self, mut span: Span, to: SyntaxContext) -> Span {
        while span.from_expansion() && span.ctxt() != to {
            span = self.expn_data(self.outer_expn(span.ctxt())).call_site;
        }
        span
    }

    fn adjust(&self, ctxt: &mut SyntaxContext, expn_id: ExpnId) -> Option<ExpnId> {
        let mut scope = None;
        while !self.is_descendant_of(expn_id, self.outer_expn(*ctxt)) {
            scope = Some(self.remove_mark(ctxt));
        }
        scope
    }

    fn apply_mark(&mut self, ctxt: SyntaxContext, expn_id: ExpnId) -> SyntaxContext {
        assert_ne!(expn_id, ExpnId::root());
        self.apply_mark_with_transparency(
            ctxt, expn_id, self.expn_data(expn_id).default_transparency
        )
    }

    fn apply_mark_with_transparency(&mut self, ctxt: SyntaxContext, expn_id: ExpnId,
                                    transparency: Transparency) -> SyntaxContext {
        assert_ne!(expn_id, ExpnId::root());
        if transparency == Transparency::Opaque {
            return self.apply_mark_internal(ctxt, expn_id, transparency);
        }

        let call_site_ctxt = self.expn_data(expn_id).call_site.ctxt();
        let mut call_site_ctxt = if transparency == Transparency::SemiTransparent {
            self.modern(call_site_ctxt)
        } else {
            self.modern_and_legacy(call_site_ctxt)
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
        &mut self, ctxt: SyntaxContext, expn_id: ExpnId, transparency: Transparency
    ) -> SyntaxContext {
        let syntax_context_data = &mut self.syntax_context_data;
        let mut opaque = syntax_context_data[ctxt.0 as usize].opaque;
        let mut opaque_and_semitransparent =
            syntax_context_data[ctxt.0 as usize].opaque_and_semitransparent;

        if transparency >= Transparency::Opaque {
            let parent = opaque;
            opaque = *self.syntax_context_map.entry((parent, expn_id, transparency))
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
            opaque_and_semitransparent =
                    *self.syntax_context_map.entry((parent, expn_id, transparency))
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
    let (len, to_update) = HygieneData::with(|data| (
        data.syntax_context_data.len(),
        data.syntax_context_data.iter().rev()
            .take_while(|scdata| scdata.dollar_crate_name == kw::DollarCrate).count()
    ));
    // The callback must be called from outside of the `HygieneData` lock,
    // since it will try to acquire it too.
    let range_to_update = len - to_update .. len;
    let names: Vec<_> =
        range_to_update.clone().map(|idx| get_name(SyntaxContext::from_u32(idx as u32))).collect();
    HygieneData::with(|data| range_to_update.zip(names.into_iter()).for_each(|(idx, name)| {
        data.syntax_context_data[idx].dollar_crate_name = name;
    }))
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

    /// Extend a syntax context with a given expansion and default transparency for that expansion.
    pub fn apply_mark(self, expn_id: ExpnId) -> SyntaxContext {
        HygieneData::with(|data| data.apply_mark(self, expn_id))
    }

    /// Extend a syntax context with a given expansion and transparency.
    pub fn apply_mark_with_transparency(self, expn_id: ExpnId, transparency: Transparency)
                                        -> SyntaxContext {
        HygieneData::with(|data| data.apply_mark_with_transparency(self, expn_id, transparency))
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
        HygieneData::with(|data| data.remove_mark(self))
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

    /// Like `SyntaxContext::adjust`, but also modernizes `self`.
    pub fn modernize_and_adjust(&mut self, expn_id: ExpnId) -> Option<ExpnId> {
        HygieneData::with(|data| {
            *self = data.modern(*self);
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
            let mut glob_ctxt = data.modern(glob_span.ctxt());
            while !data.is_descendant_of(expn_id, data.outer_expn(glob_ctxt)) {
                scope = Some(data.remove_mark(&mut glob_ctxt));
                if data.remove_mark(self) != scope.unwrap() {
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
    pub fn reverse_glob_adjust(&mut self, expn_id: ExpnId, glob_span: Span)
                               -> Option<Option<ExpnId>> {
        HygieneData::with(|data| {
            if data.adjust(self, expn_id).is_some() {
                return None;
            }

            let mut glob_ctxt = data.modern(glob_span.ctxt());
            let mut marks = Vec::new();
            while !data.is_descendant_of(expn_id, data.outer_expn(glob_ctxt)) {
                marks.push(data.remove_mark(&mut glob_ctxt));
            }

            let scope = marks.last().cloned();
            while let Some(mark) = marks.pop() {
                *self = data.apply_mark(*self, mark);
            }
            Some(scope)
        })
    }

    pub fn hygienic_eq(self, other: SyntaxContext, expn_id: ExpnId) -> bool {
        HygieneData::with(|data| {
            let mut self_modern = data.modern(self);
            data.adjust(&mut self_modern, expn_id);
            self_modern == data.modern(other)
        })
    }

    #[inline]
    pub fn modern(self) -> SyntaxContext {
        HygieneData::with(|data| data.modern(self))
    }

    #[inline]
    pub fn modern_and_legacy(self) -> SyntaxContext {
        HygieneData::with(|data| data.modern_and_legacy(self))
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

    /// `ctxt.outer_expn_with_data()` is equivalent to but faster than
    /// `{ let outer = ctxt.outer_expn(); (outer, outer.expn_data()) }`.
    #[inline]
    pub fn outer_expn_with_data(self) -> (ExpnId, ExpnData) {
        HygieneData::with(|data| {
            let outer = data.outer_expn(self);
            (outer, data.expn_data(outer).clone())
        })
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
        HygieneData::with(|data| {
            let expn_id = data.fresh_expn(Some(expn_data));
            self.with_ctxt(data.apply_mark(SyntaxContext::root(), expn_id))
        })
    }
}

/// A subset of properties from both macro definition and macro call available through global data.
/// Avoid using this if you have access to the original definition or call structures.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
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
    /// Transparency used by `apply_mark` for the expansion with this expansion data by default.
    pub default_transparency: Transparency,
    /// List of #[unstable]/feature-gated features that the macro is allowed to use
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
}

impl ExpnData {
    /// Constructs expansion data with default properties.
    pub fn default(kind: ExpnKind, call_site: Span, edition: Edition) -> ExpnData {
        ExpnData {
            kind,
            parent: ExpnId::root(),
            call_site,
            def_site: DUMMY_SP,
            default_transparency: Transparency::SemiTransparent,
            allow_internal_unstable: None,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            edition,
        }
    }

    pub fn allow_unstable(kind: ExpnKind, call_site: Span, edition: Edition,
                          allow_internal_unstable: Lrc<[Symbol]>) -> ExpnData {
        ExpnData {
            allow_internal_unstable: Some(allow_internal_unstable),
            ..ExpnData::default(kind, call_site, edition)
        }
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        if let ExpnKind::Root = self.kind { true } else { false }
    }
}

/// Expansion kind.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum ExpnKind {
    /// No expansion, aka root expansion. Only `ExpnId::root()` has this kind.
    Root,
    /// Expansion produced by a macro.
    /// FIXME: Some code injected by the compiler before HIR lowering also gets this kind.
    Macro(MacroKind, Symbol),
    /// Desugaring done by the compiler during HIR lowering.
    Desugaring(DesugaringKind)
}

impl ExpnKind {
    pub fn descr(&self) -> Symbol {
        match *self {
            ExpnKind::Root => kw::PathRoot,
            ExpnKind::Macro(_, descr) => descr,
            ExpnKind::Desugaring(kind) => Symbol::intern(kind.descr()),
        }
    }
}

/// The kind of macro invocation or definition.
#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
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

    pub fn article(self) -> &'static str {
        match self {
            MacroKind::Attr => "an",
            _ => "a",
        }
    }
}

/// The kind of compiler desugaring.
#[derive(Clone, Copy, PartialEq, Debug, RustcEncodable, RustcDecodable)]
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
    ForLoop,
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
            DesugaringKind::ForLoop => "`for` loop",
        }
    }
}

impl Encodable for ExpnId {
    fn encode<E: Encoder>(&self, _: &mut E) -> Result<(), E::Error> {
        Ok(()) // FIXME(jseyfried) intercrate hygiene
    }
}

impl Decodable for ExpnId {
    fn decode<D: Decoder>(_: &mut D) -> Result<Self, D::Error> {
        Ok(ExpnId::root()) // FIXME(jseyfried) intercrate hygiene
    }
}
