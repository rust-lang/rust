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
use symbol::Symbol;

use serialize::{Encodable, Decodable, Encoder, Decoder};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

/// A SyntaxContext represents a chain of macro expansions (represented by marks).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxContext(u32);

#[derive(Copy, Clone)]
pub struct SyntaxContextData {
    pub outer_mark: Mark,
    pub prev_ctxt: SyntaxContext,
}

/// A mark is a unique id associated with a macro expansion.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default, RustcEncodable, RustcDecodable)]
pub struct Mark(u32);

impl Mark {
    pub fn fresh() -> Self {
        HygieneData::with(|data| {
            data.marks.push(None);
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

    pub fn expn_info(self) -> Option<ExpnInfo> {
        HygieneData::with(|data| data.marks[self.0 as usize].clone())
    }

    pub fn set_expn_info(self, info: ExpnInfo) {
        HygieneData::with(|data| data.marks[self.0 as usize] = Some(info))
    }
}

struct HygieneData {
    marks: Vec<Option<ExpnInfo>>,
    syntax_contexts: Vec<SyntaxContextData>,
    markings: HashMap<(SyntaxContext, Mark), SyntaxContext>,
}

impl HygieneData {
    fn new() -> Self {
        HygieneData {
            marks: vec![None],
            syntax_contexts: vec![SyntaxContextData {
                outer_mark: Mark::root(),
                prev_ctxt: SyntaxContext::empty(),
            }],
            markings: HashMap::new(),
        }
    }

    fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        thread_local! {
            static HYGIENE_DATA: RefCell<HygieneData> = RefCell::new(HygieneData::new());
        }
        HYGIENE_DATA.with(|data| f(&mut *data.borrow_mut()))
    }
}

pub fn clear_markings() {
    HygieneData::with(|data| data.markings = HashMap::new());
}

impl SyntaxContext {
    pub const fn empty() -> Self {
        SyntaxContext(0)
    }

    pub fn data(self) -> SyntaxContextData {
        HygieneData::with(|data| data.syntax_contexts[self.0 as usize])
    }

    /// Extend a syntax context with a given mark
    pub fn apply_mark(self, mark: Mark) -> SyntaxContext {
        // Applying the same mark twice is a no-op
        let ctxt_data = self.data();
        if mark == ctxt_data.outer_mark {
            return ctxt_data.prev_ctxt;
        }

        HygieneData::with(|data| {
            let syntax_contexts = &mut data.syntax_contexts;
            *data.markings.entry((self, mark)).or_insert_with(|| {
                syntax_contexts.push(SyntaxContextData {
                    outer_mark: mark,
                    prev_ctxt: self,
                });
                SyntaxContext(syntax_contexts.len() as u32 - 1)
            })
        })
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
#[derive(Clone, Hash, Debug)]
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

#[derive(Clone, Hash, Debug)]
pub struct NameAndSpan {
    /// The format with which the macro was invoked.
    pub format: ExpnFormat,
    /// Whether the macro is allowed to use #[unstable]/feature-gated
    /// features internally without forcing the whole crate to opt-in
    /// to them.
    pub allow_internal_unstable: bool,
    /// The span of the macro definition itself. The macro may not
    /// have a sensible definition span (e.g. something defined
    /// completely inside libsyntax) in which case this is None.
    pub span: Option<Span>
}

impl NameAndSpan {
    pub fn name(&self) -> Symbol {
        match self.format {
            ExpnFormat::MacroAttribute(s) |
            ExpnFormat::MacroBang(s) |
            ExpnFormat::CompilerDesugaring(s) => s,
        }
    }
}

/// The source of expansion.
#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub enum ExpnFormat {
    /// e.g. #[derive(...)] <item>
    MacroAttribute(Symbol),
    /// e.g. `format!()`
    MacroBang(Symbol),
    /// Desugaring done by the compiler during HIR lowering.
    CompilerDesugaring(Symbol)
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
