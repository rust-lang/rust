// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Machinery for hygienic macros, inspired by the `MTWT[1]` paper.
//!
//! `[1]` Matthew Flatt, Ryan Culpepper, David Darais, and Robert Bruce Findler. 2012.
//! *Macros that work together: Compile-time bindings, partial expansion,
//! and definition contexts*. J. Funct. Program. 22, 2 (March 2012), 181-216.
//! DOI=10.1017/S0956796812000093 <https://doi.org/10.1017/S0956796812000093>

use GLOBALS;
use Span;
use edition::Edition;
use symbol::{Ident, Symbol};

use serialize::{Encodable, Decodable, Encoder, Decoder};
use std::collections::HashMap;
use rustc_data_structures::fx::FxHashSet;
use std::fmt;

/// A SyntaxContext represents a chain of macro expansions (represented by marks).
#[derive(Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct SyntaxContext(pub(super) u32);

#[derive(Copy, Clone, Debug)]
pub struct SyntaxContextData {
    pub outer_mark: Mark,
    pub prev_ctxt: SyntaxContext,
    pub modern: SyntaxContext,
}

/// A mark is a unique id associated with a macro expansion.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Mark(u32);

#[derive(Debug)]
struct MarkData {
    parent: Mark,
    transparency: Transparency,
    is_builtin: bool,
    expn_info: Option<ExpnInfo>,
}

/// A property of a macro expansion that determines how identifiers
/// produced by that expansion are resolved.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Transparency {
    /// Identifier produced by a transparent expansion is always resolved at call-site.
    /// Call-site spans in procedural macros, hygiene opt-out in `macro` should use this.
    /// (Not used yet.)
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
            data.marks.push(MarkData {
                parent,
                // By default expansions behave like `macro_rules`.
                transparency: Transparency::SemiTransparent,
                is_builtin: false,
                expn_info: None,
            });
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
    pub fn expn_info(self) -> Option<ExpnInfo> {
        HygieneData::with(|data| data.marks[self.0 as usize].expn_info.clone())
    }

    #[inline]
    pub fn set_expn_info(self, info: ExpnInfo) {
        HygieneData::with(|data| {
            let old_info = &mut data.marks[self.0 as usize].expn_info;
            if let Some(old_info) = old_info {
                panic!("expansion info is reset for the mark {}\nold: {:#?}\nnew: {:#?}",
                       self.0, old_info, info);
            }
            *old_info = Some(info);
        })
    }

    pub fn modern(mut self) -> Mark {
        HygieneData::with(|data| {
            while data.marks[self.0 as usize].transparency != Transparency::Opaque {
                self = data.marks[self.0 as usize].parent;
            }
            self
        })
    }

    #[inline]
    pub fn transparency(self) -> Transparency {
        assert_ne!(self, Mark::root());
        HygieneData::with(|data| data.marks[self.0 as usize].transparency)
    }

    #[inline]
    pub fn set_transparency(self, transparency: Transparency) {
        assert_ne!(self, Mark::root());
        HygieneData::with(|data| data.marks[self.0 as usize].transparency = transparency)
    }

    #[inline]
    pub fn is_builtin(self) -> bool {
        HygieneData::with(|data| data.marks[self.0 as usize].is_builtin)
    }

    #[inline]
    pub fn set_is_builtin(self, is_builtin: bool) {
        HygieneData::with(|data| data.marks[self.0 as usize].is_builtin = is_builtin)
    }

    pub fn is_descendant_of(mut self, ancestor: Mark) -> bool {
        HygieneData::with(|data| {
            while self != ancestor {
                if self == Mark::root() {
                    return false;
                }
                self = data.marks[self.0 as usize].parent;
            }
            true
        })
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
            let mut a_path = FxHashSet::<Mark>();
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
}

#[derive(Debug)]
pub struct HygieneData {
    marks: Vec<MarkData>,
    syntax_contexts: Vec<SyntaxContextData>,
    markings: HashMap<(SyntaxContext, Mark), SyntaxContext>,
    gensym_to_ctxt: HashMap<Symbol, Span>,
    default_edition: Edition,
}

impl HygieneData {
    pub fn new() -> Self {
        HygieneData {
            marks: vec![MarkData {
                parent: Mark::root(),
                // If the root is opaque, then loops searching for an opaque mark
                // will automatically stop after reaching it.
                transparency: Transparency::Opaque,
                is_builtin: true,
                expn_info: None,
            }],
            syntax_contexts: vec![SyntaxContextData {
                outer_mark: Mark::root(),
                prev_ctxt: SyntaxContext(0),
                modern: SyntaxContext(0),
            }],
            markings: HashMap::new(),
            gensym_to_ctxt: HashMap::new(),
            default_edition: Edition::Edition2015,
        }
    }

    fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        GLOBALS.with(|globals| f(&mut *globals.hygiene_data.borrow_mut()))
    }
}

pub fn default_edition() -> Edition {
    HygieneData::with(|data| data.default_edition)
}

pub fn set_default_edition(edition: Edition) {
    HygieneData::with(|data| data.default_edition = edition);
}

pub fn clear_markings() {
    HygieneData::with(|data| data.markings = HashMap::new());
}

impl SyntaxContext {
    pub const fn empty() -> Self {
        SyntaxContext(0)
    }

    // Allocate a new SyntaxContext with the given ExpnInfo. This is used when
    // deserializing Spans from the incr. comp. cache.
    // FIXME(mw): This method does not restore MarkData::parent or
    // SyntaxContextData::prev_ctxt or SyntaxContextData::modern. These things
    // don't seem to be used after HIR lowering, so everything should be fine
    // as long as incremental compilation does not kick in before that.
    pub fn allocate_directly(expansion_info: ExpnInfo) -> Self {
        HygieneData::with(|data| {
            data.marks.push(MarkData {
                parent: Mark::root(),
                transparency: Transparency::SemiTransparent,
                is_builtin: false,
                expn_info: Some(expansion_info),
            });

            let mark = Mark(data.marks.len() as u32 - 1);

            data.syntax_contexts.push(SyntaxContextData {
                outer_mark: mark,
                prev_ctxt: SyntaxContext::empty(),
                modern: SyntaxContext::empty(),
            });
            SyntaxContext(data.syntax_contexts.len() as u32 - 1)
        })
    }

    /// Extend a syntax context with a given mark
    pub fn apply_mark(self, mark: Mark) -> SyntaxContext {
        if mark.transparency() == Transparency::Opaque {
            return self.apply_mark_internal(mark);
        }

        let call_site_ctxt =
            mark.expn_info().map_or(SyntaxContext::empty(), |info| info.call_site.ctxt()).modern();
        if call_site_ctxt == SyntaxContext::empty() {
            return self.apply_mark_internal(mark);
        }

        // Otherwise, `mark` is a macros 1.0 definition and the call site is in a
        // macros 2.0 expansion, i.e. a macros 1.0 invocation is in a macros 2.0 definition.
        //
        // In this case, the tokens from the macros 1.0 definition inherit the hygiene
        // at their invocation. That is, we pretend that the macros 1.0 definition
        // was defined at its invocation (i.e. inside the macros 2.0 definition)
        // so that the macros 2.0 definition remains hygienic.
        //
        // See the example at `test/run-pass/hygiene/legacy_interaction.rs`.
        let mut ctxt = call_site_ctxt;
        for mark in self.marks() {
            ctxt = ctxt.apply_mark_internal(mark);
        }
        ctxt.apply_mark_internal(mark)
    }

    fn apply_mark_internal(self, mark: Mark) -> SyntaxContext {
        HygieneData::with(|data| {
            let syntax_contexts = &mut data.syntax_contexts;
            let mut modern = syntax_contexts[self.0 as usize].modern;
            if data.marks[mark.0 as usize].transparency == Transparency::Opaque {
                modern = *data.markings.entry((modern, mark)).or_insert_with(|| {
                    let len = syntax_contexts.len() as u32;
                    syntax_contexts.push(SyntaxContextData {
                        outer_mark: mark,
                        prev_ctxt: modern,
                        modern: SyntaxContext(len),
                    });
                    SyntaxContext(len)
                });
            }

            *data.markings.entry((self, mark)).or_insert_with(|| {
                syntax_contexts.push(SyntaxContextData {
                    outer_mark: mark,
                    prev_ctxt: self,
                    modern,
                });
                SyntaxContext(syntax_contexts.len() as u32 - 1)
            })
        })
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
        HygieneData::with(|data| {
            let outer_mark = data.syntax_contexts[self.0 as usize].outer_mark;
            *self = data.syntax_contexts[self.0 as usize].prev_ctxt;
            outer_mark
        })
    }

    pub fn marks(mut self) -> Vec<Mark> {
        HygieneData::with(|data| {
            let mut marks = Vec::new();
            while self != SyntaxContext::empty() {
                marks.push(data.syntax_contexts[self.0 as usize].outer_mark);
                self = data.syntax_contexts[self.0 as usize].prev_ctxt;
            }
            marks.reverse();
            marks
        })
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
    /// or `None` if we privacy check as usual (i.e. not w.r.t. a macro definition scope).
    pub fn adjust(&mut self, expansion: Mark) -> Option<Mark> {
        let mut scope = None;
        while !expansion.is_descendant_of(self.outer()) {
            scope = Some(self.remove_mark());
        }
        scope
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
    pub fn glob_adjust(&mut self, expansion: Mark, mut glob_ctxt: SyntaxContext)
                       -> Option<Option<Mark>> {
        let mut scope = None;
        while !expansion.is_descendant_of(glob_ctxt.outer()) {
            scope = Some(glob_ctxt.remove_mark());
            if self.remove_mark() != scope.unwrap() {
                return None;
            }
        }
        if self.adjust(expansion).is_some() {
            return None;
        }
        Some(scope)
    }

    /// Undo `glob_adjust` if possible:
    ///
    /// ```rust
    /// if let Some(privacy_checking_scope) = self.reverse_glob_adjust(expansion, glob_ctxt) {
    ///     assert!(self.glob_adjust(expansion, glob_ctxt) == Some(privacy_checking_scope));
    /// }
    /// ```
    pub fn reverse_glob_adjust(&mut self, expansion: Mark, mut glob_ctxt: SyntaxContext)
                               -> Option<Option<Mark>> {
        if self.adjust(expansion).is_some() {
            return None;
        }

        let mut marks = Vec::new();
        while !expansion.is_descendant_of(glob_ctxt.outer()) {
            marks.push(glob_ctxt.remove_mark());
        }

        let scope = marks.last().cloned();
        while let Some(mark) = marks.pop() {
            *self = self.apply_mark(mark);
        }
        Some(scope)
    }

    #[inline]
    pub fn modern(self) -> SyntaxContext {
        HygieneData::with(|data| data.syntax_contexts[self.0 as usize].modern)
    }

    #[inline]
    pub fn outer(self) -> Mark {
        HygieneData::with(|data| data.syntax_contexts[self.0 as usize].outer_mark)
    }
}

impl fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Extra information for tracking spans of macro and syntax sugar expansion
#[derive(Clone, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ExpnInfo {
    /// The location of the actual macro invocation or syntax sugar , e.g.
    /// `let x = foo!();` or `if let Some(y) = x {}`
    ///
    /// This may recursively refer to other macro invocations, e.g. if
    /// `foo!()` invoked `bar!()` internally, and there was an
    /// expression inside `bar!`; the call_site of the expression in
    /// the expansion would point to the `bar!` invocation; that
    /// call_site span would have its own ExpnInfo, with the call_site
    /// pointing to the `foo!` invocation.
    pub call_site: Span,
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g. something defined
    /// completely inside libsyntax) in which case this is None.
    /// This span serves only informational purpose and is not used for resolution.
    pub def_site: Option<Span>,
    /// The format with which the macro was invoked.
    pub format: ExpnFormat,
    /// Whether the macro is allowed to use #[unstable]/feature-gated
    /// features internally without forcing the whole crate to opt-in
    /// to them.
    pub allow_internal_unstable: bool,
    /// Whether the macro is allowed to use `unsafe` internally
    /// even if the user crate has `#![forbid(unsafe_code)]`.
    pub allow_internal_unsafe: bool,
    /// Edition of the crate in which the macro is defined.
    pub edition: Edition,
}

/// The source of expansion.
#[derive(Clone, Hash, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum ExpnFormat {
    /// e.g. #[derive(...)] <item>
    MacroAttribute(Symbol),
    /// e.g. `format!()`
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
    DotFill,
    QuestionMark,
    Catch,
    /// Desugaring of an `impl Trait` in return type position
    /// to an `existential type Foo: Trait;` + replacing the
    /// `impl Trait` with `Foo`.
    ExistentialReturnType,
    Async,
}

impl CompilerDesugaringKind {
    pub fn name(self) -> Symbol {
        Symbol::intern(match self {
            CompilerDesugaringKind::Async => "async",
            CompilerDesugaringKind::DotFill => "...",
            CompilerDesugaringKind::QuestionMark => "?",
            CompilerDesugaringKind::Catch => "do catch",
            CompilerDesugaringKind::ExistentialReturnType => "existential type",
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

impl Symbol {
    pub fn from_ident(ident: Ident) -> Symbol {
        HygieneData::with(|data| {
            let gensym = ident.name.gensymed();
            data.gensym_to_ctxt.insert(gensym, ident.span);
            gensym
        })
    }

    pub fn to_ident(self) -> Ident {
        HygieneData::with(|data| {
            match data.gensym_to_ctxt.get(&self) {
                Some(&span) => Ident::new(self.interned(), span),
                None => Ident::with_empty_ctxt(self),
            }
        })
    }
}
