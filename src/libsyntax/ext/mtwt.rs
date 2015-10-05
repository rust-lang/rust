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
/// the mark_memo and rename_memo fields are side-tables
/// that ensure that adding the same mark to the same context
/// gives you back the same context as before. This shouldn't
/// change the semantics--everything here is immutable--but
/// it should cut down on memory use *a lot*; applying a mark
/// to a tree containing 50 identifiers would otherwise generate
/// 50 new contexts
pub struct SCTable {
    table: RefCell<Vec<SyntaxContext_>>,
    mark_memo: RefCell<HashMap<(SyntaxContext,Mrk),SyntaxContext>>,
    // The pair (Name,SyntaxContext) is actually one Ident, but it needs to be hashed and
    // compared as pair (name, ctxt) and not as an Ident
    rename_memo: RefCell<HashMap<(SyntaxContext,(Name,SyntaxContext),Name),SyntaxContext>>,
}

#[derive(PartialEq, RustcEncodable, RustcDecodable, Hash, Debug, Copy, Clone)]
pub enum SyntaxContext_ {
    EmptyCtxt,
    Mark (Mrk,SyntaxContext),
    /// flattening the name and syntaxcontext into the rename...
    /// HIDDEN INVARIANTS:
    /// 1) the first name in a Rename node
    /// can only be a programmer-supplied name.
    /// 2) Every Rename node with a given Name in the
    /// "to" slot must have the same name and context
    /// in the "from" slot. In essence, they're all
    /// pointers to a single "rename" event node.
    Rename (Ident,Name,SyntaxContext),
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
    let key = (ctxt, m);
    *table.mark_memo.borrow_mut().entry(key).or_insert_with(|| {
        SyntaxContext(idx_push(&mut *table.table.borrow_mut(), Mark(m, ctxt)))
    })
}

/// Extend a syntax context with a given rename
pub fn apply_rename(id: Ident, to:Name,
                  ctxt: SyntaxContext) -> SyntaxContext {
    with_sctable(|table| apply_rename_internal(id, to, ctxt, table))
}

/// Extend a syntax context with a given rename and sctable (explicit memoization)
fn apply_rename_internal(id: Ident,
                       to: Name,
                       ctxt: SyntaxContext,
                       table: &SCTable) -> SyntaxContext {
    let key = (ctxt, (id.name, id.ctxt), to);

    *table.rename_memo.borrow_mut().entry(key).or_insert_with(|| {
            SyntaxContext(idx_push(&mut *table.table.borrow_mut(), Rename(id, to, ctxt)))
    })
}

/// Apply a list of renamings to a context
// if these rename lists get long, it would make sense
// to consider memoizing this fold. This may come up
// when we add hygiene to item names.
pub fn apply_renames(renames: &RenameList, ctxt: SyntaxContext) -> SyntaxContext {
    renames.iter().fold(ctxt, |ctxt, &(from, to)| {
        apply_rename(from, to, ctxt)
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
        mark_memo: RefCell::new(HashMap::new()),
        rename_memo: RefCell::new(HashMap::new()),
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
        *table.mark_memo.borrow_mut() = HashMap::new();
        *table.rename_memo.borrow_mut() = HashMap::new();
    });
    with_resolve_table_mut(|table| *table = HashMap::new());
}

/// Reset the tables to their initial state
pub fn reset_tables() {
    with_sctable(|table| {
        *table.table.borrow_mut() = vec!(EmptyCtxt, IllegalCtxt);
        *table.mark_memo.borrow_mut() = HashMap::new();
        *table.rename_memo.borrow_mut() = HashMap::new();
    });
    with_resolve_table_mut(|table| *table = HashMap::new());
}

/// Add a value to the end of a vec, return its index
fn idx_push<T>(vec: &mut Vec<T>, val: T) -> u32 {
    vec.push(val);
    (vec.len() - 1) as u32
}

/// Resolve a syntax object to a name, per MTWT.
pub fn resolve(id: Ident) -> Name {
    with_sctable(|sctable| {
        with_resolve_table_mut(|resolve_table| {
            resolve_internal(id, sctable, resolve_table)
        })
    })
}

type ResolveTable = HashMap<(Name,SyntaxContext),Name>;

// okay, I admit, putting this in TLS is not so nice:
// fetch the SCTable from TLS, create one if it doesn't yet exist.
fn with_resolve_table_mut<T, F>(op: F) -> T where
    F: FnOnce(&mut ResolveTable) -> T,
{
    thread_local!(static RESOLVE_TABLE_KEY: RefCell<ResolveTable> = {
        RefCell::new(HashMap::new())
    });

    RESOLVE_TABLE_KEY.with(move |slot| op(&mut *slot.borrow_mut()))
}

/// Resolve a syntax object to a name, per MTWT.
/// adding memoization to resolve 500+ seconds in resolve for librustc (!)
fn resolve_internal(id: Ident,
                    table: &SCTable,
                    resolve_table: &mut ResolveTable) -> Name {
    let key = (id.name, id.ctxt);

    match resolve_table.get(&key) {
        Some(&name) => return name,
        None => {}
    }

    let resolved = {
        let result = (*table.table.borrow())[id.ctxt.0 as usize];
        match result {
            EmptyCtxt => id.name,
            // ignore marks here:
            Mark(_,subctxt) =>
                resolve_internal(Ident::new(id.name, subctxt),
                                 table, resolve_table),
            // do the rename if necessary:
            Rename(Ident{name, ctxt}, toname, subctxt) => {
                let resolvedfrom =
                    resolve_internal(Ident::new(name, ctxt),
                                     table, resolve_table);
                let resolvedthis =
                    resolve_internal(Ident::new(id.name, subctxt),
                                     table, resolve_table);
                if (resolvedthis == resolvedfrom)
                    && (marksof_internal(ctxt, resolvedthis, table)
                        == marksof_internal(subctxt, resolvedthis, table)) {
                    toname
                } else {
                    resolvedthis
                }
            }
            IllegalCtxt => panic!("expected resolvable context, got IllegalCtxt")
        }
    };
    resolve_table.insert(key, resolved);
    resolved
}

/// Compute the marks associated with a syntax context.
pub fn marksof(ctxt: SyntaxContext, stopname: Name) -> Vec<Mrk> {
    with_sctable(|table| marksof_internal(ctxt, stopname, table))
}

// the internal function for computing marks
// it's not clear to me whether it's better to use a [] mutable
// vector or a cons-list for this.
fn marksof_internal(ctxt: SyntaxContext,
                    stopname: Name,
                    table: &SCTable) -> Vec<Mrk> {
    let mut result = Vec::new();
    let mut loopvar = ctxt;
    loop {
        let table_entry = (*table.table.borrow())[loopvar.0 as usize];
        match table_entry {
            EmptyCtxt => {
                return result;
            },
            Mark(mark, tl) => {
                xor_push(&mut result, mark);
                loopvar = tl;
            },
            Rename(_,name,tl) => {
                // see MTWT for details on the purpose of the stopname.
                // short version: it prevents duplication of effort.
                if name == stopname {
                    return result;
                } else {
                    loopvar = tl;
                }
            }
            IllegalCtxt => panic!("expected resolvable context, got IllegalCtxt")
        }
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

/// Push a name... unless it matches the one on top, in which
/// case pop and discard (so two of the same marks cancel)
fn xor_push(marks: &mut Vec<Mrk>, mark: Mrk) {
    if (!marks.is_empty()) && (*marks.last().unwrap() == mark) {
        marks.pop().unwrap();
    } else {
        marks.push(mark);
    }
}

#[cfg(test)]
mod tests {
    use self::TestSC::*;
    use ast::{EMPTY_CTXT, Ident, Mrk, Name, SyntaxContext};
    use super::{resolve, xor_push, apply_mark_internal, new_sctable_internal};
    use super::{apply_rename_internal, apply_renames, marksof_internal, resolve_internal};
    use super::{SCTable, EmptyCtxt, Mark, Rename, IllegalCtxt};
    use std::collections::HashMap;

    #[test]
    fn xorpush_test () {
        let mut s = Vec::new();
        xor_push(&mut s, 14);
        assert_eq!(s.clone(), [14]);
        xor_push(&mut s, 14);
        assert_eq!(s.clone(), []);
        xor_push(&mut s, 14);
        assert_eq!(s.clone(), [14]);
        xor_push(&mut s, 15);
        assert_eq!(s.clone(), [14, 15]);
        xor_push(&mut s, 16);
        assert_eq!(s.clone(), [14, 15, 16]);
        xor_push(&mut s, 16);
        assert_eq!(s.clone(), [14, 15]);
        xor_push(&mut s, 15);
        assert_eq!(s.clone(), [14]);
    }

    fn id(n: u32, s: SyntaxContext) -> Ident {
        Ident::new(Name(n), s)
    }

    // because of the SCTable, I now need a tidy way of
    // creating syntax objects. Sigh.
    #[derive(Clone, PartialEq, Debug)]
    enum TestSC {
        M(Mrk),
        R(Ident,Name)
    }

    // unfold a vector of TestSC values into a SCTable,
    // returning the resulting index
    fn unfold_test_sc(tscs : Vec<TestSC> , tail: SyntaxContext, table: &SCTable)
        -> SyntaxContext {
        tscs.iter().rev().fold(tail, |tail : SyntaxContext, tsc : &TestSC|
                  {match *tsc {
                      M(mrk) => apply_mark_internal(mrk,tail,table),
                      R(ident,name) => apply_rename_internal(ident,name,tail,table)}})
    }

    // gather a SyntaxContext back into a vector of TestSCs
    fn refold_test_sc(mut sc: SyntaxContext, table : &SCTable) -> Vec<TestSC> {
        let mut result = Vec::new();
        loop {
            let table = table.table.borrow();
            match (*table)[sc.0 as usize] {
                EmptyCtxt => {return result;},
                Mark(mrk,tail) => {
                    result.push(M(mrk));
                    sc = tail;
                    continue;
                },
                Rename(id,name,tail) => {
                    result.push(R(id,name));
                    sc = tail;
                    continue;
                }
                IllegalCtxt => panic!("expected resolvable context, got IllegalCtxt")
            }
        }
    }

    #[test]
    fn test_unfold_refold(){
        let mut t = new_sctable_internal();

        let test_sc = vec!(M(3),R(id(101,EMPTY_CTXT),Name(14)),M(9));
        assert_eq!(unfold_test_sc(test_sc.clone(),EMPTY_CTXT,&mut t),SyntaxContext(4));
        {
            let table = t.table.borrow();
            assert!((*table)[2] == Mark(9,EMPTY_CTXT));
            assert!((*table)[3] == Rename(id(101,EMPTY_CTXT),Name(14),SyntaxContext(2)));
            assert!((*table)[4] == Mark(3,SyntaxContext(3)));
        }
        assert_eq!(refold_test_sc(SyntaxContext(4),&t),test_sc);
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
    fn test_marksof () {
        let stopname = Name(242);
        let name1 = Name(243);
        let mut t = new_sctable_internal();
        assert_eq!(marksof_internal (EMPTY_CTXT,stopname,&t),Vec::new());
        // FIXME #5074: ANF'd to dodge nested calls
        { let ans = unfold_marks(vec!(4,98),EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans,stopname,&t), [4, 98]);}
        // does xoring work?
        { let ans = unfold_marks(vec!(5,5,16),EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans,stopname,&t), [16]);}
        // does nested xoring work?
        { let ans = unfold_marks(vec!(5,10,10,5,16),EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans, stopname,&t), [16]);}
        // rename where stop doesn't match:
        { let chain = vec!(M(9),
                        R(id(name1.0,
                             apply_mark_internal (4, EMPTY_CTXT,&mut t)),
                          Name(100101102)),
                        M(14));
         let ans = unfold_test_sc(chain,EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans, stopname, &t), [9, 14]);}
        // rename where stop does match
        { let name1sc = apply_mark_internal(4, EMPTY_CTXT, &mut t);
         let chain = vec!(M(9),
                       R(id(name1.0, name1sc),
                         stopname),
                       M(14));
         let ans = unfold_test_sc(chain,EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans, stopname, &t), [9]); }
    }


    #[test]
    fn resolve_tests () {
        let a = 40;
        let mut t = new_sctable_internal();
        let mut rt = HashMap::new();
        // - ctxt is MT
        assert_eq!(resolve_internal(id(a,EMPTY_CTXT),&mut t, &mut rt),Name(a));
        // - simple ignored marks
        { let sc = unfold_marks(vec!(1,2,3),EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),Name(a));}
        // - orthogonal rename where names don't match
        { let sc = unfold_test_sc(vec!(R(id(50,EMPTY_CTXT),Name(51)),M(12)),EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),Name(a));}
        // - rename where names do match, but marks don't
        { let sc1 = apply_mark_internal(1,EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(vec!(R(id(a,sc1),Name(50)),
                                   M(1),
                                   M(2)),
                                 EMPTY_CTXT,&mut t);
        assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), Name(a));}
        // - rename where names and marks match
        { let sc1 = unfold_test_sc(vec!(M(1),M(2)),EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(vec!(R(id(a,sc1),Name(50)),M(1),M(2)),EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), Name(50)); }
        // - rename where names and marks match by literal sharing
        { let sc1 = unfold_test_sc(vec!(M(1),M(2)),EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(vec!(R(id(a,sc1),Name(50))),sc1,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), Name(50)); }
        // - two renames of the same var.. can only happen if you use
        // local-expand to prevent the inner binding from being renamed
        // during the rename-pass caused by the first:
        println!("about to run bad test");
        { let sc = unfold_test_sc(vec!(R(id(a,EMPTY_CTXT),Name(50)),
                                    R(id(a,EMPTY_CTXT),Name(51))),
                                  EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), Name(51)); }
        // the simplest double-rename:
        { let a_to_a50 = apply_rename_internal(id(a,EMPTY_CTXT),Name(50),EMPTY_CTXT,&mut t);
         let a50_to_a51 = apply_rename_internal(id(a,a_to_a50),Name(51),a_to_a50,&mut t);
         assert_eq!(resolve_internal(id(a,a50_to_a51),&mut t, &mut rt),Name(51));
         // mark on the outside doesn't stop rename:
         let sc = apply_mark_internal(9,a50_to_a51,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),Name(51));
         // but mark on the inside does:
         let a50_to_a51_b = unfold_test_sc(vec!(R(id(a,a_to_a50),Name(51)),
                                              M(9)),
                                           a_to_a50,
                                           &mut t);
         assert_eq!(resolve_internal(id(a,a50_to_a51_b),&mut t, &mut rt),Name(50));}
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

    #[test]
    fn resolve_table_hashing_tests() {
        let mut t = new_sctable_internal();
        let mut rt = HashMap::new();
        assert_eq!(rt.len(),0);
        resolve_internal(id(30,EMPTY_CTXT),&mut t, &mut rt);
        assert_eq!(rt.len(),1);
        resolve_internal(id(39,EMPTY_CTXT),&mut t, &mut rt);
        assert_eq!(rt.len(),2);
        resolve_internal(id(30,EMPTY_CTXT),&mut t, &mut rt);
        assert_eq!(rt.len(),2);
    }

    #[test]
    fn new_resolves_test() {
        let renames = vec!((Ident::with_empty_ctxt(Name(23)),Name(24)),
                           (Ident::with_empty_ctxt(Name(29)),Name(29)));
        let new_ctxt1 = apply_renames(&renames,EMPTY_CTXT);
        assert_eq!(resolve(Ident::new(Name(23),new_ctxt1)),Name(24));
        assert_eq!(resolve(Ident::new(Name(29),new_ctxt1)),Name(29));
    }
}
