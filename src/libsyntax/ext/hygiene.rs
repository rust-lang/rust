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

use ast::NodeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

/// A SyntaxContext represents a chain of macro expansions (represented by marks).
#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Default)]
pub struct SyntaxContext(u32);

#[derive(Copy, Clone)]
pub struct SyntaxContextData {
    pub outer_mark: Mark,
    pub prev_ctxt: SyntaxContext,
}

/// A mark is a unique id associated with a macro expansion.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Mark(u32);

impl Mark {
    pub fn fresh() -> Self {
        HygieneData::with(|data| {
            let next_mark = Mark(data.next_mark.0 + 1);
            ::std::mem::replace(&mut data.next_mark, next_mark)
        })
    }

    /// The mark of the theoretical expansion that generates freshly parsed, unexpanded AST.
    pub fn root() -> Self {
        Mark(0)
    }

    pub fn from_placeholder_id(id: NodeId) -> Self {
        Mark(id.as_u32())
    }

    pub fn as_placeholder_id(self) -> NodeId {
        NodeId::from_u32(self.0)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

struct HygieneData {
    syntax_contexts: Vec<SyntaxContextData>,
    markings: HashMap<(SyntaxContext, Mark), SyntaxContext>,
    next_mark: Mark,
}

impl HygieneData {
    fn new() -> Self {
        HygieneData {
            syntax_contexts: vec![SyntaxContextData {
                outer_mark: Mark::root(),
                prev_ctxt: SyntaxContext::empty(),
            }],
            markings: HashMap::new(),
            next_mark: Mark(1),
        }
    }

    fn with<T, F: FnOnce(&mut HygieneData) -> T>(f: F) -> T {
        thread_local! {
            static HYGIENE_DATA: RefCell<HygieneData> = RefCell::new(HygieneData::new());
        }
        HYGIENE_DATA.with(|data| f(&mut *data.borrow_mut()))
    }
}

pub fn reset_hygiene_data() {
    HygieneData::with(|data| *data = HygieneData::new())
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

    /// If `ident` is macro expanded, return the source ident from the macro definition
    /// and the mark of the expansion that created the macro definition.
    pub fn source(self) -> (Self /* source context */, Mark /* source macro */) {
         let macro_def_ctxt = self.data().prev_ctxt.data();
         (macro_def_ctxt.prev_ctxt, macro_def_ctxt.outer_mark)
    }
}

impl fmt::Debug for SyntaxContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}
