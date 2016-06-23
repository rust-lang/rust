// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Machinery for hygienic macros, as described in the MTWT[1] paper.
//!
//! [1] Matthew Flatt, Ryan Culpepper, David Darais, and Robert Bruce Findler.
//! 2012. *Macros that work together: Compile-time bindings, partial expansion,
//! and definition contexts*. J. Funct. Program. 22, 2 (March 2012), 181-216.
//! DOI=10.1017/S0956796812000093 http://dx.doi.org/10.1017/S0956796812000093

pub use self::SyntaxContext_::*;

use ast::{Ident, Mrk, Name, SyntaxContext};

use std::cell::RefCell;
use std::collections::HashMap;

/// The SCTable contains a table of SyntaxContext_'s. It
/// represents a flattened tree structure, to avoid having
/// managed pointers everywhere (that caused an ICE).
/// the `marks` and `renames` fields are side-tables
/// that ensure that adding the same mark to the same context
/// gives you back the same context as before. This should cut
/// down on memory use *a lot*; applying a mark to a tree containing
/// 50 identifiers would otherwise generate 50 new contexts.
pub struct SCTable {
    table: RefCell<Vec<SyntaxContext_>>,
    marks: RefCell<HashMap<(SyntaxContext,Mrk),SyntaxContext>>,
    renames: RefCell<HashMap<Name,SyntaxContext>>,
}

#[derive(PartialEq, RustcEncodable, RustcDecodable, Hash, Debug, Copy, Clone)]
pub enum SyntaxContext_ {
    EmptyCtxt,
    Mark (Mrk,SyntaxContext),
    Rename (Name),
    /// actually, IllegalCtxt may not be necessary.
    IllegalCtxt
}

/// A list of ident->name renamings
pub type RenameList = Vec<(Ident, Name)>;

/// Extend a syntax context with a given mark
pub fn apply_mark(m: Mrk, ctxt: SyntaxContext) -> SyntaxContext {
    with_sctable(|table| apply_mark_internal(m, ctxt, table))
}

/// Extend a syntax context with a given mark and sctable (explicit memoization)
fn apply_mark_internal(m: Mrk, ctxt: SyntaxContext, table: &SCTable) -> SyntaxContext {
    let ctxts = &mut *table.table.borrow_mut();
    match ctxts[ctxt.0 as usize] {
        // Applying the same mark twice is a no-op.
        Mark(outer_mark, prev_ctxt) if outer_mark == m => return prev_ctxt,
        _ => *table.marks.borrow_mut().entry((ctxt, m)).or_insert_with(|| {
            SyntaxContext(idx_push(ctxts, Mark(m, ctxt)))
        }),
    }
}

/// Extend a syntax context with a given rename
pub fn apply_rename(from: Ident, to: Name, ident: Ident) -> Ident {
    with_sctable(|table| apply_rename_internal(from, to, ident, table))
}

/// Extend a syntax context with a given rename and sctable (explicit memoization)
fn apply_rename_internal(from: Ident, to: Name, ident: Ident, table: &SCTable) -> Ident {
    if (ident.name, ident.ctxt) != (from.name, from.ctxt) {
        return ident;
    }
    let ctxt = *table.renames.borrow_mut().entry(to).or_insert_with(|| {
        SyntaxContext(idx_push(&mut *table.table.borrow_mut(), Rename(to)))
    });
    Ident { ctxt: ctxt, ..ident }
}

/// Apply a list of renamings to a context
// if these rename lists get long, it would make sense
// to consider memoizing this fold. This may come up
// when we add hygiene to item names.
pub fn apply_renames(renames: &RenameList, ident: Ident) -> Ident {
    renames.iter().fold(ident, |ident, &(from, to)| {
        apply_rename(from, to, ident)
    })
}

/// Fetch the SCTable from TLS, create one if it doesn't yet exist.
pub fn with_sctable<T, F>(op: F) -> T where
    F: FnOnce(&SCTable) -> T,
{
    thread_local!(static SCTABLE_KEY: SCTable = new_sctable_internal());
    SCTABLE_KEY.with(move |slot| op(slot))
}

// Make a fresh syntax context table with EmptyCtxt in slot zero
// and IllegalCtxt in slot one.
fn new_sctable_internal() -> SCTable {
    SCTable {
        table: RefCell::new(vec!(EmptyCtxt, IllegalCtxt)),
        marks: RefCell::new(HashMap::new()),
        renames: RefCell::new(HashMap::new()),
    }
}

/// Print out an SCTable for debugging
pub fn display_sctable(table: &SCTable) {
    error!("SC table:");
    for (idx,val) in table.table.borrow().iter().enumerate() {
        error!("{:4} : {:?}",idx,val);
    }
}

/// Clear the tables from TLD to reclaim memory.
pub fn clear_tables() {
    with_sctable(|table| {
        *table.table.borrow_mut() = Vec::new();
        *table.marks.borrow_mut() = HashMap::new();
        *table.renames.borrow_mut() = HashMap::new();
    });
}

/// Reset the tables to their initial state
pub fn reset_tables() {
    with_sctable(|table| {
        *table.table.borrow_mut() = vec!(EmptyCtxt, IllegalCtxt);
        *table.marks.borrow_mut() = HashMap::new();
        *table.renames.borrow_mut() = HashMap::new();
    });
}

/// Add a value to the end of a vec, return its index
fn idx_push<T>(vec: &mut Vec<T>, val: T) -> u32 {
    vec.push(val);
    (vec.len() - 1) as u32
}

/// Resolve a syntax object to a name, per MTWT.
pub fn resolve(id: Ident) -> Name {
    with_sctable(|sctable| {
        resolve_internal(id, sctable)
    })
}

/// Resolve a syntax object to a name, per MTWT.
/// adding memoization to resolve 500+ seconds in resolve for librustc (!)
fn resolve_internal(id: Ident, table: &SCTable) -> Name {
    match table.table.borrow()[id.ctxt.0 as usize] {
        EmptyCtxt => id.name,
        // ignore marks here:
        Mark(_, subctxt) => resolve_internal(Ident::new(id.name, subctxt), table),
        Rename(name) => name,
        IllegalCtxt => panic!("expected resolvable context, got IllegalCtxt")
    }
}

/// Return the outer mark for a context with a mark at the outside.
/// FAILS when outside is not a mark.
pub fn outer_mark(ctxt: SyntaxContext) -> Mrk {
    with_sctable(|sctable| {
        match (*sctable.table.borrow())[ctxt.0 as usize] {
            Mark(mrk, _) => mrk,
            _ => panic!("can't retrieve outer mark when outside is not a mark")
        }
    })
}

#[cfg(test)]
mod tests {
    use ast::{EMPTY_CTXT, Ident, Mrk, Name, SyntaxContext};
    use super::{resolve, apply_mark_internal, new_sctable_internal};
    use super::{SCTable, Mark};

    fn id(n: u32, s: SyntaxContext) -> Ident {
        Ident::new(Name(n), s)
    }

    // extend a syntax context with a sequence of marks given
    // in a vector. v[0] will be the outermost mark.
    fn unfold_marks(mrks: Vec<Mrk> , tail: SyntaxContext, table: &SCTable)
                    -> SyntaxContext {
        mrks.iter().rev().fold(tail, |tail:SyntaxContext, mrk:&Mrk|
                   {apply_mark_internal(*mrk,tail,table)})
    }

    #[test] fn unfold_marks_test() {
        let mut t = new_sctable_internal();

        assert_eq!(unfold_marks(vec!(3,7),EMPTY_CTXT,&mut t),SyntaxContext(3));
        {
            let table = t.table.borrow();
            assert!((*table)[2] == Mark(7,EMPTY_CTXT));
            assert!((*table)[3] == Mark(3,SyntaxContext(2)));
        }
    }

    #[test]
    fn mtwt_resolve_test(){
        let a = 40;
        assert_eq!(resolve(id(a,EMPTY_CTXT)),Name(a));
    }

    #[test]
    fn hashing_tests () {
        let mut t = new_sctable_internal();
        assert_eq!(apply_mark_internal(12,EMPTY_CTXT,&mut t),SyntaxContext(2));
        assert_eq!(apply_mark_internal(13,EMPTY_CTXT,&mut t),SyntaxContext(3));
        // using the same one again should result in the same index:
        assert_eq!(apply_mark_internal(12,EMPTY_CTXT,&mut t),SyntaxContext(2));
        // I'm assuming that the rename table will behave the same....
    }
}
