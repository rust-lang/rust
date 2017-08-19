// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Machinery for hygienic macros, inspired by the MTWT[1] paper.
//!
//! [1] Matthew Flatt, Ryan Culpepper, David Darais, and Robert Bruce Findler.
//! 2012. *Macros that work together: Compile-time bindings, partial expansion,
//! and definition contexts*. J. Funct. Program. 22, 2 (March 2012), 181-216.
//! DOI=10.1017/S0956796812000093 http://dx.doi.org/10.1017/S0956796812000093

use Span;
use symbol::{Ident, Symbol};

use serialize::{Decoder, Encoder, UseSpecializedDecodable, UseSpecializedEncodable};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// A SyntaxContext represents a chain of macro expansions (represented by marks).
#[derive(Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub struct SyntaxContext(pub(super) u32);

#[derive(Copy, Clone, Default, RustcEncodable, RustcDecodable)]
pub struct SyntaxContextData {
    pub outer_mark: Mark,
    pub prev_ctxt: SyntaxContext,
    pub modern: SyntaxContext,
}

/// A mark is a unique id associated with a macro expansion.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Mark(u32);

#[derive(Clone, Default, RustcEncodable, RustcDecodable)]
struct MarkData {
    parent: Mark,
    modern: bool,
    expn_info: Option<ExpnInfo>,
}

impl Mark {
    pub fn fresh(parent: Mark) -> Self {
        HygieneData::with(|data| {
            data.marks.push(MarkData { parent: parent, modern: false, expn_info: None });
            Mark(data.marks.len() as u32 - 1)
        })
    }

    /// The mark of the theoretical expansion that generates freshly parsed, unexpanded AST.
    pub fn root() -> Self {
        Mark(0)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn from_u32(raw: u32) -> Mark {
        Mark(raw)
    }

    pub fn translate(&self, offset: u32) -> Mark {
        if self.0 != 0 {
            Mark(self.0 + offset)
        } else {
            Mark(self.0)
        }
    }

    pub fn expn_info(self) -> Option<ExpnInfo> {
        HygieneData::with(|data| data.marks[self.0 as usize].expn_info.clone())
    }

    pub fn set_expn_info(self, info: ExpnInfo) {
        HygieneData::with(|data| data.marks[self.0 as usize].expn_info = Some(info))
    }

    pub fn modern(mut self) -> Mark {
        HygieneData::with(|data| {
            loop {
                if self == Mark::root() || data.marks[self.0 as usize].modern {
                    return self;
                }
                self = data.marks[self.0 as usize].parent;
            }
        })
    }

    pub fn is_modern(self) -> bool {
        HygieneData::with(|data| data.marks[self.0 as usize].modern)
    }

    pub fn set_modern(self) {
        HygieneData::with(|data| data.marks[self.0 as usize].modern = true)
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
}

pub struct HygieneData {
    marks: Vec<MarkData>,
    syntax_contexts: Vec<SyntaxContextData>,
    markings: HashMap<(SyntaxContext, Mark), SyntaxContext>,
    gensym_to_ctxt: HashMap<Symbol, SyntaxContext>,
    used_marks: Vec<Mark>,
    used_syntax_contexts: Vec<SyntaxContext>,
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct HygieneDataMap {
    marks: HashMap<Mark, MarkData>,
    syntax_contexts: HashMap<SyntaxContext, SyntaxContextData>,
    gensym_to_ctxt: HashMap<Symbol, SyntaxContext>,
}

thread_local! {
    static HYGIENE_DATA: RefCell<HygieneData> = RefCell::new(HygieneData::new());
}

impl HygieneData {
    fn new() -> Self {
        HygieneData {
            marks: vec![MarkData::default()],
            syntax_contexts: vec![SyntaxContextData::default()],
            markings: HashMap::new(),
            gensym_to_ctxt: HashMap::new(),
            used_marks: Vec::new(),
            used_syntax_contexts: Vec::new(),
        }
    }

    fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        HYGIENE_DATA.with(|data| f(&mut *data.borrow_mut()))
    }

    pub fn safe_with<T, F: FnOnce(&HygieneData) -> T>(f: F) -> T {
        HYGIENE_DATA.with(|data| f(&*data.borrow()))
    }

    pub fn to_map(&self) -> HygieneDataMap {
        let mut marks = HashMap::new();
        let mut syntax_contexts = HashMap::new();

        let mut mark_queue: VecDeque<_> = self.used_marks.iter().cloned().collect();
        let mut ctxt_queue: VecDeque<_> = self.used_syntax_contexts.iter().cloned().collect();
        ctxt_queue.extend(self.gensym_to_ctxt.values());
        let gensym_to_ctxt = self.gensym_to_ctxt.clone();

        let mut visited_marks = HashSet::new();
        let mut visited_ctxts = HashSet::new();

        while !(mark_queue.is_empty() && ctxt_queue.is_empty()) {
            let next_mark = mark_queue.pop_front().and_then(|mark|
                // skip default mark and already visited marks
                if visited_marks.contains(&mark) || mark.0 == 0 {
                    None
                } else {
                    visited_marks.insert(mark);
                    Some(mark)
                });
            let next_ctxt = ctxt_queue.pop_front().and_then(|ctxt|
                // skip default context and already visited contexts
                if visited_ctxts.contains(&ctxt) || ctxt.0 == 0 {
                    None
                } else {
                    visited_ctxts.insert(ctxt);
                    Some(ctxt)
                });

            if let Some(mark) = next_mark {
                let data = &self.marks[mark.0 as usize];

                mark_queue.push_back(data.parent);
                if let Some(ref info) = data.expn_info {
                    ctxt_queue.push_back(info.call_site.ctxt);

                    if let Some(span) = info.callee.span {
                        ctxt_queue.push_back(span.ctxt);
                    }
                }

                marks.insert(mark, data.clone());
            }

            if let Some(ctxt) = next_ctxt {
                let data = self.syntax_contexts[ctxt.0 as usize];

                mark_queue.push_back(data.outer_mark);
                ctxt_queue.push_back(data.prev_ctxt);
                ctxt_queue.push_back(data.modern);

                syntax_contexts.insert(ctxt, data);
            }
        }

        HygieneDataMap {
            marks,
            syntax_contexts,
            gensym_to_ctxt,
        }
    }
}

pub fn clear_markings() {
    HygieneData::with(|data| data.markings = HashMap::new());
}

fn register_mark_use(mark: Mark) {
    HygieneData::with(|data| if !data.used_marks.contains(&mark) {
        data.used_marks.push(mark);
    });
}

fn register_syntax_context_use(ctxt: SyntaxContext) {
    HygieneData::with(|data| if !data.used_syntax_contexts.contains(&ctxt) {
        data.used_syntax_contexts.push(ctxt)
    });
}

/// Holds information about a HygieneData imported from another crate.
/// See `imported_hygiene_data()` in `rustc_metadata` for more information.
#[derive(Default)]
pub struct ImportedHygieneData {
    /// Map an external crate's syntax contexts to the current crate's.
    ctxt_map: HashMap<SyntaxContext, SyntaxContext>,
    /// Map an external crate's marks to the current crate's.
    mark_map: HashMap<Mark, Mark>,
}

impl ImportedHygieneData {
    fn insert_ctxt(&mut self, external: SyntaxContext, target: SyntaxContext) {
        assert!(!self.ctxt_map.contains_key(&external));
        self.ctxt_map.insert(external, target);
    }

    fn insert_mark(&mut self, external: Mark, target: Mark) {
        assert!(!self.mark_map.contains_key(&external));
        self.mark_map.insert(external, target);
    }

    pub fn translate_ctxt(&self, external: SyntaxContext) -> SyntaxContext {
        if external.0 != 0 {
            self.ctxt_map[&external]
        } else {
            external
        }
    }

    pub fn translate_mark(&self, external: Mark) -> Mark {
        if external.0 != 0 {
            self.mark_map[&external]
        } else {
            external
        }
    }

    pub fn translate_span(&self, external: Span) -> Span {
        Span {
            lo: external.lo,
            hi: external.hi,
            ctxt: self.translate_ctxt(external.ctxt),
        }
    }

    fn translate_mark_data(&self, data: MarkData) -> MarkData {
        MarkData {
            parent: self.translate_mark(data.parent),
            modern: data.modern,
            expn_info: data.expn_info.as_ref().map(|info| {
                ExpnInfo {
                    call_site: self.translate_span(info.call_site),
                    callee: NameAndSpan {
                        format: info.callee.format.clone(),
                        allow_internal_unstable: info.callee.allow_internal_unstable,
                        allow_internal_unsafe: info.callee.allow_internal_unsafe,
                        span: info.callee.span.map(|span| self.translate_span(span)),
                    },
                }
            }),
        }
    }

    fn translate_ctxt_data(&self, data: SyntaxContextData) -> SyntaxContextData {
        SyntaxContextData {
            outer_mark: self.translate_mark(data.outer_mark),
            prev_ctxt: self.translate_ctxt(data.prev_ctxt),
            modern: self.translate_ctxt(data.modern),
        }
    }
}

pub fn extend_hygiene_data(extend_with: HygieneDataMap) -> ImportedHygieneData {
    HygieneData::with(move |data| {
        let mut imported_map = ImportedHygieneData::default();
        let mark_offset = data.marks.len() as u32;
        let ctxt_offset = data.syntax_contexts.len() as u32;

        let HygieneDataMap {
            mut marks,
            mut syntax_contexts,
            mut gensym_to_ctxt,
        } = extend_with;

        let marks: Vec<_> = marks
            .drain()
            .enumerate()
            .map(|(index_offset, (mark, data))| {
                let index_offset = index_offset as u32;
                imported_map.insert_mark(mark, Mark(mark_offset + index_offset));
                data
            })
            .collect();

        let syntax_contexts: Vec<_> = syntax_contexts
            .drain()
            .enumerate()
            .map(|(index_offset, (ctxt, data))| {
                let index_offset = index_offset as u32;
                imported_map.insert_ctxt(ctxt, SyntaxContext(ctxt_offset + index_offset));
                data
            })
            .collect();

        for mark in marks {
            data.marks.push(imported_map.translate_mark_data(mark));
        }

        for ctxt in syntax_contexts {
            data.syntax_contexts.push(imported_map.translate_ctxt_data(ctxt));
        }

        data.gensym_to_ctxt
            .extend(gensym_to_ctxt
                        .drain()
                        .map(|(symbol, ctxt)| (symbol, imported_map.translate_ctxt(ctxt))));

        imported_map
    })
}

impl SyntaxContext {
    pub fn from_u32(raw: u32) -> SyntaxContext {
        SyntaxContext(raw)
    }

    pub fn translate(&self, offset: u32) -> SyntaxContext {
        if self.0 != 0 {
            SyntaxContext(self.0 + offset)
        } else {
            SyntaxContext(self.0)
        }
    }

    pub const fn empty() -> Self {
        SyntaxContext(0)
    }

    /// Extend a syntax context with a given mark
    pub fn apply_mark(self, mark: Mark) -> SyntaxContext {
        HygieneData::with(|data| {
            let syntax_contexts = &mut data.syntax_contexts;
            let mut modern = syntax_contexts[self.0 as usize].modern;
            if data.marks[mark.0 as usize].modern {
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

    pub fn remove_mark(&mut self) -> Mark {
        HygieneData::with(|data| {
            let outer_mark = data.syntax_contexts[self.0 as usize].outer_mark;
            *self = data.syntax_contexts[self.0 as usize].prev_ctxt;
            outer_mark
        })
    }

    /// Adjust this context for resolution in a scope created by the given expansion.
    /// For example, consider the following three resolutions of `f`:
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
    /// For example,
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

    pub fn modern(self) -> SyntaxContext {
        HygieneData::with(|data| data.syntax_contexts[self.0 as usize].modern)
    }

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
    /// Information about the expansion.
    pub callee: NameAndSpan
}

#[derive(Clone, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct NameAndSpan {
    /// The format with which the macro was invoked.
    pub format: ExpnFormat,
    /// Whether the macro is allowed to use #[unstable]/feature-gated
    /// features internally without forcing the whole crate to opt-in
    /// to them.
    pub allow_internal_unstable: bool,
    /// Whether the macro is allowed to use `unsafe` internally
    /// even if the user crate has `#![forbid(unsafe_code)]`.
    pub allow_internal_unsafe: bool,
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g. something defined
    /// completely inside libsyntax) in which case this is None.
    pub span: Option<Span>
}

impl NameAndSpan {
    pub fn name(&self) -> Symbol {
        match self.format {
            ExpnFormat::MacroAttribute(s) |
            ExpnFormat::MacroBang(s) => s,
            ExpnFormat::CompilerDesugaring(ref kind) => kind.as_symbol(),
        }
    }
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

/// The kind of compiler desugaring.
#[derive(Clone, Hash, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable)]
pub enum CompilerDesugaringKind {
    BackArrow,
    DotFill,
    QuestionMark,
}

impl CompilerDesugaringKind {
    pub fn as_symbol(&self) -> Symbol {
        use CompilerDesugaringKind::*;
        let s = match *self {
            BackArrow => "<-",
            DotFill => "...",
            QuestionMark => "?",
        };
        Symbol::intern(s)
    }
}

impl UseSpecializedDecodable for SyntaxContext {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<SyntaxContext, D::Error> {
        d.read_u32().map(|u| SyntaxContext(u))
    }
}

impl UseSpecializedEncodable for SyntaxContext {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        register_syntax_context_use(*self);
        s.emit_u32(self.0)
    }
}

impl UseSpecializedDecodable for Mark {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<Mark, D::Error> {
        d.read_u32().map(Mark::from_u32)
    }
}

impl UseSpecializedEncodable for Mark {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        register_mark_use(*self);
        s.emit_u32(self.0)
    }
}

impl Symbol {
    pub fn from_ident(ident: Ident) -> Symbol {
        HygieneData::with(|data| {
            let gensym = ident.name.gensymed();
            data.gensym_to_ctxt.insert(gensym, ident.ctxt);
            gensym
        })
    }

    pub fn to_ident(self) -> Ident {
        HygieneData::with(|data| {
            match data.gensym_to_ctxt.get(&self) {
                Some(&ctxt) => Ident { name: self.interned(), ctxt: ctxt },
                None => Ident::with_empty_ctxt(self),
            }
        })
    }
}
