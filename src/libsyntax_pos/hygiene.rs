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
// This explains why `HygieneData`, `SyntaxContext` and `Mark` have interfaces
// with a certain amount of redundancy in them. For example,
// `SyntaxContext::outer_expn_info` combines `SyntaxContext::outer` and
// `Mark::expn_info` so that two `HygieneData` accesses can be performed within
// a single `HygieneData::with` call.
//
// It also explains why many functions appear in `HygieneData` and again in
// `SyntaxContext` or `Mark`. For example, `HygieneData::outer` and
// `SyntaxContext::outer` do the same thing, but the former is for use within a
// `HygieneData::with` call while the latter is for use outside such a call.
// When modifying this file it is important to understand this distinction,
// because getting it wrong can lead to nested `HygieneData::with` calls that
// trigger runtime aborts. (Fortunately these are obvious and easy to fix.)

use crate::GLOBALS;
use crate::Span;
use crate::edition::Edition;
use crate::symbol::{kw, Symbol};

use serialize::{Encodable, Decodable, Encoder, Decoder};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use std::{fmt, mem};

/// A SyntaxContext represents a chain of macro expansions (represented by marks).
#[derive(Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct SyntaxContext(u32);

#[derive(Copy, Clone, Debug)]
struct SyntaxContextData {
    outer_mark: Mark,
    transparency: Transparency,
    prev_ctxt: SyntaxContext,
    /// This context, but with all transparent and semi-transparent marks filtered away.
    opaque: SyntaxContext,
    /// This context, but with all transparent marks filtered away.
    opaque_and_semitransparent: SyntaxContext,
    /// Name of the crate to which `$crate` with this context would resolve.
    dollar_crate_name: Symbol,
}

/// A mark is a unique ID associated with a macro expansion.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Mark(u32);

#[derive(Clone, Debug)]
struct MarkData {
    parent: Mark,
    expn_info: Option<ExpnInfo>,
}

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

impl Mark {
    pub fn fresh(parent: Mark) -> Self {
        HygieneData::with(|data| {
            data.marks.push(MarkData { parent, expn_info: None });
            Mark(data.marks.len() as u32 - 1)
        })
    }

    /// The mark of the theoretical expansion that generates freshly parsed, unexpanded AST.
    #[inline]
    pub fn root() -> Self {
        Mark(0)
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn from_u32(raw: u32) -> Mark {
        Mark(raw)
    }

    #[inline]
    pub fn parent(self) -> Mark {
        HygieneData::with(|data| data.marks[self.0 as usize].parent)
    }

    #[inline]
    pub fn expn_info(self) -> Option<ExpnInfo> {
        HygieneData::with(|data| data.expn_info(self).cloned())
    }

    #[inline]
    pub fn set_expn_info(self, info: ExpnInfo) {
        HygieneData::with(|data| data.marks[self.0 as usize].expn_info = Some(info))
    }

    pub fn is_descendant_of(self, ancestor: Mark) -> bool {
        HygieneData::with(|data| data.is_descendant_of(self, ancestor))
    }

    /// `mark.outer_is_descendant_of(ctxt)` is equivalent to but faster than
    /// `mark.is_descendant_of(ctxt.outer())`.
    pub fn outer_is_descendant_of(self, ctxt: SyntaxContext) -> bool {
        HygieneData::with(|data| data.is_descendant_of(self, data.outer(ctxt)))
    }

    /// Computes a mark such that both input marks are descendants of (or equal to) the returned
    /// mark. That is, the following holds:
    ///
    /// ```rust
    /// let la = least_ancestor(a, b);
    /// assert!(a.is_descendant_of(la))
    /// assert!(b.is_descendant_of(la))
    /// ```
    pub fn least_ancestor(mut a: Mark, mut b: Mark) -> Mark {
        HygieneData::with(|data| {
            // Compute the path from a to the root
            let mut a_path = FxHashSet::<Mark>::default();
            while a != Mark::root() {
                a_path.insert(a);
                a = data.marks[a.0 as usize].parent;
            }

            // While the path from b to the root hasn't intersected, move up the tree
            while !a_path.contains(&b) {
                b = data.marks[b.0 as usize].parent;
            }

            b
        })
    }

    // Used for enabling some compatibility fallback in resolve.
    #[inline]
    pub fn looks_like_proc_macro_derive(self) -> bool {
        HygieneData::with(|data| {
            if data.default_transparency(self) == Transparency::Opaque {
                if let Some(expn_info) = &data.marks[self.0 as usize].expn_info {
                    if let ExpnFormat::MacroAttribute(name) = expn_info.format {
                        if name.as_str().starts_with("derive(") {
                            return true;
                        }
                    }
                }
            }
            false
        })
    }
}

#[derive(Debug)]
crate struct HygieneData {
    marks: Vec<MarkData>,
    syntax_contexts: Vec<SyntaxContextData>,
    markings: FxHashMap<(SyntaxContext, Mark, Transparency), SyntaxContext>,
}

impl HygieneData {
    crate fn new() -> Self {
        HygieneData {
            marks: vec![MarkData {
                parent: Mark::root(),
                expn_info: None,
            }],
            syntax_contexts: vec![SyntaxContextData {
                outer_mark: Mark::root(),
                transparency: Transparency::Opaque,
                prev_ctxt: SyntaxContext(0),
                opaque: SyntaxContext(0),
                opaque_and_semitransparent: SyntaxContext(0),
                dollar_crate_name: kw::DollarCrate,
            }],
            markings: FxHashMap::default(),
        }
    }

    fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        GLOBALS.with(|globals| f(&mut *globals.hygiene_data.borrow_mut()))
    }

    fn expn_info(&self, mark: Mark) -> Option<&ExpnInfo> {
        self.marks[mark.0 as usize].expn_info.as_ref()
    }

    fn is_descendant_of(&self, mut mark: Mark, ancestor: Mark) -> bool {
        while mark != ancestor {
            if mark == Mark::root() {
                return false;
            }
            mark = self.marks[mark.0 as usize].parent;
        }
        true
    }

    fn default_transparency(&self, mark: Mark) -> Transparency {
        self.marks[mark.0 as usize].expn_info.as_ref().map_or(
            Transparency::SemiTransparent, |einfo| einfo.default_transparency
        )
    }

    fn modern(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_contexts[ctxt.0 as usize].opaque
    }

    fn modern_and_legacy(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_contexts[ctxt.0 as usize].opaque_and_semitransparent
    }

    fn outer(&self, ctxt: SyntaxContext) -> Mark {
        self.syntax_contexts[ctxt.0 as usize].outer_mark
    }

    fn transparency(&self, ctxt: SyntaxContext) -> Transparency {
        self.syntax_contexts[ctxt.0 as usize].transparency
    }

    fn prev_ctxt(&self, ctxt: SyntaxContext) -> SyntaxContext {
        self.syntax_contexts[ctxt.0 as usize].prev_ctxt
    }

    fn remove_mark(&self, ctxt: &mut SyntaxContext) -> Mark {
        let outer_mark = self.syntax_contexts[ctxt.0 as usize].outer_mark;
        *ctxt = self.prev_ctxt(*ctxt);
        outer_mark
    }

    fn marks(&self, mut ctxt: SyntaxContext) -> Vec<(Mark, Transparency)> {
        let mut marks = Vec::new();
        while ctxt != SyntaxContext::empty() {
            let outer_mark = self.outer(ctxt);
            let transparency = self.transparency(ctxt);
            let prev_ctxt = self.prev_ctxt(ctxt);
            marks.push((outer_mark, transparency));
            ctxt = prev_ctxt;
        }
        marks.reverse();
        marks
    }

    fn walk_chain(&self, mut span: Span, to: SyntaxContext) -> Span {
        while span.ctxt() != crate::NO_EXPANSION && span.ctxt() != to {
            if let Some(info) = self.expn_info(self.outer(span.ctxt())) {
                span = info.call_site;
            } else {
                break;
            }
        }
        span
    }

    fn adjust(&self, ctxt: &mut SyntaxContext, expansion: Mark) -> Option<Mark> {
        let mut scope = None;
        while !self.is_descendant_of(expansion, self.outer(*ctxt)) {
            scope = Some(self.remove_mark(ctxt));
        }
        scope
    }

    fn apply_mark(&mut self, ctxt: SyntaxContext, mark: Mark) -> SyntaxContext {
        assert_ne!(mark, Mark::root());
        self.apply_mark_with_transparency(ctxt, mark, self.default_transparency(mark))
    }

    fn apply_mark_with_transparency(&mut self, ctxt: SyntaxContext, mark: Mark,
                                    transparency: Transparency) -> SyntaxContext {
        assert_ne!(mark, Mark::root());
        if transparency == Transparency::Opaque {
            return self.apply_mark_internal(ctxt, mark, transparency);
        }

        let call_site_ctxt =
            self.expn_info(mark).map_or(SyntaxContext::empty(), |info| info.call_site.ctxt());
        let mut call_site_ctxt = if transparency == Transparency::SemiTransparent {
            self.modern(call_site_ctxt)
        } else {
            self.modern_and_legacy(call_site_ctxt)
        };

        if call_site_ctxt == SyntaxContext::empty() {
            return self.apply_mark_internal(ctxt, mark, transparency);
        }

        // Otherwise, `mark` is a macros 1.0 definition and the call site is in a
        // macros 2.0 expansion, i.e., a macros 1.0 invocation is in a macros 2.0 definition.
        //
        // In this case, the tokens from the macros 1.0 definition inherit the hygiene
        // at their invocation. That is, we pretend that the macros 1.0 definition
        // was defined at its invocation (i.e., inside the macros 2.0 definition)
        // so that the macros 2.0 definition remains hygienic.
        //
        // See the example at `test/run-pass/hygiene/legacy_interaction.rs`.
        for (mark, transparency) in self.marks(ctxt) {
            call_site_ctxt = self.apply_mark_internal(call_site_ctxt, mark, transparency);
        }
        self.apply_mark_internal(call_site_ctxt, mark, transparency)
    }

    fn apply_mark_internal(&mut self, ctxt: SyntaxContext, mark: Mark, transparency: Transparency)
                           -> SyntaxContext {
        let syntax_contexts = &mut self.syntax_contexts;
        let mut opaque = syntax_contexts[ctxt.0 as usize].opaque;
        let mut opaque_and_semitransparent =
            syntax_contexts[ctxt.0 as usize].opaque_and_semitransparent;

        if transparency >= Transparency::Opaque {
            let prev_ctxt = opaque;
            opaque = *self.markings.entry((prev_ctxt, mark, transparency)).or_insert_with(|| {
                let new_opaque = SyntaxContext(syntax_contexts.len() as u32);
                syntax_contexts.push(SyntaxContextData {
                    outer_mark: mark,
                    transparency,
                    prev_ctxt,
                    opaque: new_opaque,
                    opaque_and_semitransparent: new_opaque,
                    dollar_crate_name: kw::DollarCrate,
                });
                new_opaque
            });
        }

        if transparency >= Transparency::SemiTransparent {
            let prev_ctxt = opaque_and_semitransparent;
            opaque_and_semitransparent =
                    *self.markings.entry((prev_ctxt, mark, transparency)).or_insert_with(|| {
                let new_opaque_and_semitransparent =
                    SyntaxContext(syntax_contexts.len() as u32);
                syntax_contexts.push(SyntaxContextData {
                    outer_mark: mark,
                    transparency,
                    prev_ctxt,
                    opaque,
                    opaque_and_semitransparent: new_opaque_and_semitransparent,
                    dollar_crate_name: kw::DollarCrate,
                });
                new_opaque_and_semitransparent
            });
        }

        let prev_ctxt = ctxt;
        *self.markings.entry((prev_ctxt, mark, transparency)).or_insert_with(|| {
            let new_opaque_and_semitransparent_and_transparent =
                SyntaxContext(syntax_contexts.len() as u32);
            syntax_contexts.push(SyntaxContextData {
                outer_mark: mark,
                transparency,
                prev_ctxt,
                opaque,
                opaque_and_semitransparent,
                dollar_crate_name: kw::DollarCrate,
            });
            new_opaque_and_semitransparent_and_transparent
        })
    }
}

pub fn clear_markings() {
    HygieneData::with(|data| data.markings = FxHashMap::default());
}

pub fn walk_chain(span: Span, to: SyntaxContext) -> Span {
    HygieneData::with(|data| data.walk_chain(span, to))
}

impl SyntaxContext {
    #[inline]
    pub const fn empty() -> Self {
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

    // Allocate a new SyntaxContext with the given ExpnInfo. This is used when
    // deserializing Spans from the incr. comp. cache.
    // FIXME(mw): This method does not restore MarkData::parent or
    // SyntaxContextData::prev_ctxt or SyntaxContextData::opaque. These things
    // don't seem to be used after HIR lowering, so everything should be fine
    // as long as incremental compilation does not kick in before that.
    pub fn allocate_directly(expansion_info: ExpnInfo) -> Self {
        HygieneData::with(|data| {
            data.marks.push(MarkData {
                parent: Mark::root(),
                expn_info: Some(expansion_info),
            });

            let mark = Mark(data.marks.len() as u32 - 1);

            data.syntax_contexts.push(SyntaxContextData {
                outer_mark: mark,
                transparency: Transparency::SemiTransparent,
                prev_ctxt: SyntaxContext::empty(),
                opaque: SyntaxContext::empty(),
                opaque_and_semitransparent: SyntaxContext::empty(),
                dollar_crate_name: kw::DollarCrate,
            });
            SyntaxContext(data.syntax_contexts.len() as u32 - 1)
        })
    }

    /// Extend a syntax context with a given mark and default transparency for that mark.
    pub fn apply_mark(self, mark: Mark) -> SyntaxContext {
        HygieneData::with(|data| data.apply_mark(self, mark))
    }

    /// Extend a syntax context with a given mark and transparency
    pub fn apply_mark_with_transparency(self, mark: Mark, transparency: Transparency)
                                        -> SyntaxContext {
        HygieneData::with(|data| data.apply_mark_with_transparency(self, mark, transparency))
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
    pub fn remove_mark(&mut self) -> Mark {
        HygieneData::with(|data| data.remove_mark(self))
    }

    pub fn marks(self) -> Vec<(Mark, Transparency)> {
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
    ///         pub fn f() {} // `f`'s `SyntaxContext` has a single `Mark` from `m`.
    ///         pub fn $f() {} // `$f`'s `SyntaxContext` is empty.
    ///     }
    ///     foo::f(); // `f`'s `SyntaxContext` has a single `Mark` from `m`
    ///     //^ Since `mod foo` is outside this expansion, `adjust` removes the mark from `f`,
    ///     //| and it resolves to `::foo::f`.
    ///     bar::f(); // `f`'s `SyntaxContext` has a single `Mark` from `m`
    ///     //^ Since `mod bar` not outside this expansion, `adjust` does not change `f`,
    ///     //| and it resolves to `::bar::f`.
    ///     bar::$f(); // `f`'s `SyntaxContext` is empty.
    ///     //^ Since `mod bar` is not outside this expansion, `adjust` does not change `$f`,
    ///     //| and it resolves to `::bar::$f`.
    /// }
    /// ```
    /// This returns the expansion whose definition scope we use to privacy check the resolution,
    /// or `None` if we privacy check as usual (i.e., not w.r.t. a macro definition scope).
    pub fn adjust(&mut self, expansion: Mark) -> Option<Mark> {
        HygieneData::with(|data| data.adjust(self, expansion))
    }

    /// Like `SyntaxContext::adjust`, but also modernizes `self`.
    pub fn modernize_and_adjust(&mut self, expansion: Mark) -> Option<Mark> {
        HygieneData::with(|data| {
            *self = data.modern(*self);
            data.adjust(self, expansion)
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
    ///         pub fn f() {} // `f`'s `SyntaxContext` has a single `Mark` from `m`.
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
    pub fn glob_adjust(&mut self, expansion: Mark, glob_span: Span) -> Option<Option<Mark>> {
        HygieneData::with(|data| {
            let mut scope = None;
            let mut glob_ctxt = data.modern(glob_span.ctxt());
            while !data.is_descendant_of(expansion, data.outer(glob_ctxt)) {
                scope = Some(data.remove_mark(&mut glob_ctxt));
                if data.remove_mark(self) != scope.unwrap() {
                    return None;
                }
            }
            if data.adjust(self, expansion).is_some() {
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
    pub fn reverse_glob_adjust(&mut self, expansion: Mark, glob_span: Span)
                               -> Option<Option<Mark>> {
        HygieneData::with(|data| {
            if data.adjust(self, expansion).is_some() {
                return None;
            }

            let mut glob_ctxt = data.modern(glob_span.ctxt());
            let mut marks = Vec::new();
            while !data.is_descendant_of(expansion, data.outer(glob_ctxt)) {
                marks.push(data.remove_mark(&mut glob_ctxt));
            }

            let scope = marks.last().cloned();
            while let Some(mark) = marks.pop() {
                *self = data.apply_mark(*self, mark);
            }
            Some(scope)
        })
    }

    pub fn hygienic_eq(self, other: SyntaxContext, mark: Mark) -> bool {
        HygieneData::with(|data| {
            let mut self_modern = data.modern(self);
            data.adjust(&mut self_modern, mark);
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
    pub fn outer(self) -> Mark {
        HygieneData::with(|data| data.outer(self))
    }

    /// `ctxt.outer_expn_info()` is equivalent to but faster than
    /// `ctxt.outer().expn_info()`.
    #[inline]
    pub fn outer_expn_info(self) -> Option<ExpnInfo> {
        HygieneData::with(|data| data.expn_info(data.outer(self)).cloned())
    }

    /// `ctxt.outer_and_expn_info()` is equivalent to but faster than
    /// `{ let outer = ctxt.outer(); (outer, outer.expn_info()) }`.
    #[inline]
    pub fn outer_and_expn_info(self) -> (Mark, Option<ExpnInfo>) {
        HygieneData::with(|data| {
            let outer = data.outer(self);
            (outer, data.expn_info(outer).cloned())
        })
    }

    pub fn dollar_crate_name(self) -> Symbol {
        HygieneData::with(|data| data.syntax_contexts[self.0 as usize].dollar_crate_name)
    }

    pub fn set_dollar_crate_name(self, dollar_crate_name: Symbol) {
        HygieneData::with(|data| {
            let prev_dollar_crate_name = mem::replace(
                &mut data.syntax_contexts[self.0 as usize].dollar_crate_name, dollar_crate_name
            );
            assert!(dollar_crate_name == prev_dollar_crate_name ||
                    prev_dollar_crate_name == kw::DollarCrate,
                    "$crate name is reset for a syntax context");
        })
    }
}

impl fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Extra information for tracking spans of macro and syntax sugar expansion
#[derive(Clone, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ExpnInfo {
    // --- The part unique to each expansion.
    /// The location of the actual macro invocation or syntax sugar , e.g.
    /// `let x = foo!();` or `if let Some(y) = x {}`
    ///
    /// This may recursively refer to other macro invocations, e.g., if
    /// `foo!()` invoked `bar!()` internally, and there was an
    /// expression inside `bar!`; the call_site of the expression in
    /// the expansion would point to the `bar!` invocation; that
    /// call_site span would have its own ExpnInfo, with the call_site
    /// pointing to the `foo!` invocation.
    pub call_site: Span,
    /// The format with which the macro was invoked.
    pub format: ExpnFormat,

    // --- The part specific to the macro/desugaring definition.
    // --- FIXME: Share it between expansions with the same definition.
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g., something defined
    /// completely inside libsyntax) in which case this is None.
    /// This span serves only informational purpose and is not used for resolution.
    pub def_site: Option<Span>,
    /// Transparency used by `apply_mark` for mark with this expansion info by default.
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

impl ExpnInfo {
    /// Constructs an expansion info with default properties.
    pub fn default(format: ExpnFormat, call_site: Span, edition: Edition) -> ExpnInfo {
        ExpnInfo {
            call_site,
            format,
            def_site: None,
            default_transparency: Transparency::SemiTransparent,
            allow_internal_unstable: None,
            allow_internal_unsafe: false,
            local_inner_macros: false,
            edition,
        }
    }

    pub fn with_unstable(format: ExpnFormat, call_site: Span, edition: Edition,
                         allow_internal_unstable: &[Symbol]) -> ExpnInfo {
        ExpnInfo {
            allow_internal_unstable: Some(allow_internal_unstable.into()),
            ..ExpnInfo::default(format, call_site, edition)
        }
    }
}

/// The source of expansion.
#[derive(Clone, Hash, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum ExpnFormat {
    /// e.g., #[derive(...)] <item>
    MacroAttribute(Symbol),
    /// e.g., `format!()`
    MacroBang(Symbol),
    /// Desugaring done by the compiler during HIR lowering.
    CompilerDesugaring(CompilerDesugaringKind)
}

impl ExpnFormat {
    pub fn name(&self) -> Symbol {
        match *self {
            ExpnFormat::MacroBang(name) | ExpnFormat::MacroAttribute(name) => name,
            ExpnFormat::CompilerDesugaring(kind) => kind.name(),
        }
    }
}

/// The kind of compiler desugaring.
#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum CompilerDesugaringKind {
    /// We desugar `if c { i } else { e }` to `match $ExprKind::Use(c) { true => i, _ => e }`.
    /// However, we do not want to blame `c` for unreachability but rather say that `i`
    /// is unreachable. This desugaring kind allows us to avoid blaming `c`.
    IfTemporary,
    QuestionMark,
    TryBlock,
    /// Desugaring of an `impl Trait` in return type position
    /// to an `existential type Foo: Trait;` and replacing the
    /// `impl Trait` with `Foo`.
    ExistentialType,
    Async,
    Await,
    ForLoop,
}

impl CompilerDesugaringKind {
    pub fn name(self) -> Symbol {
        Symbol::intern(match self {
            CompilerDesugaringKind::IfTemporary => "if",
            CompilerDesugaringKind::Async => "async",
            CompilerDesugaringKind::Await => "await",
            CompilerDesugaringKind::QuestionMark => "?",
            CompilerDesugaringKind::TryBlock => "try block",
            CompilerDesugaringKind::ExistentialType => "existential type",
            CompilerDesugaringKind::ForLoop => "for loop",
        })
    }
}

impl Encodable for SyntaxContext {
    fn encode<E: Encoder>(&self, _: &mut E) -> Result<(), E::Error> {
        Ok(()) // FIXME(jseyfried) intercrate hygiene
    }
}

impl Decodable for SyntaxContext {
    fn decode<D: Decoder>(_: &mut D) -> Result<SyntaxContext, D::Error> {
        Ok(SyntaxContext::empty()) // FIXME(jseyfried) intercrate hygiene
    }
}
