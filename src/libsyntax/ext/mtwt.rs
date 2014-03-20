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

use ast::{Ident, Mrk, Name, SyntaxContext};

use std::cell::RefCell;
use std::local_data;
use std::rc::Rc;
use std::vec::Vec;

use collections::HashMap;

// the SCTable contains a table of SyntaxContext_'s. It
// represents a flattened tree structure, to avoid having
// managed pointers everywhere (that caused an ICE).
// the mark_memo and rename_memo fields are side-tables
// that ensure that adding the same mark to the same context
// gives you back the same context as before. This shouldn't
// change the semantics--everything here is immutable--but
// it should cut down on memory use *a lot*; applying a mark
// to a tree containing 50 identifiers would otherwise generate
pub struct SCTable {
    table: RefCell<Vec<SyntaxContext_>>,
    mark_memo: RefCell<HashMap<(SyntaxContext,Mrk),SyntaxContext>>,
    rename_memo: RefCell<HashMap<(SyntaxContext,Ident,Name),SyntaxContext>>,
}

#[deriving(Eq, Encodable, Decodable, Hash)]
pub enum SyntaxContext_ {
    EmptyCtxt,
    Mark (Mrk,SyntaxContext),
    // flattening the name and syntaxcontext into the rename...
    // HIDDEN INVARIANTS:
    // 1) the first name in a Rename node
    // can only be a programmer-supplied name.
    // 2) Every Rename node with a given Name in the
    // "to" slot must have the same name and context
    // in the "from" slot. In essence, they're all
    // pointers to a single "rename" event node.
    Rename (Ident,Name,SyntaxContext),
    // actually, IllegalCtxt may not be necessary.
    IllegalCtxt
}

/// Extend a syntax context with a given mark
pub fn new_mark(m: Mrk, tail: SyntaxContext) -> SyntaxContext {
    with_sctable(|table| new_mark_internal(m, tail, table))
}

// Extend a syntax context with a given mark and table
fn new_mark_internal(m: Mrk, tail: SyntaxContext, table: &SCTable) -> SyntaxContext {
    let key = (tail, m);
    let mut mark_memo = table.mark_memo.borrow_mut();
    let new_ctxt = |_: &(SyntaxContext, Mrk)|
                   idx_push(table.table.borrow_mut().get(), Mark(m, tail));

    *mark_memo.get().find_or_insert_with(key, new_ctxt)
}

/// Extend a syntax context with a given rename
pub fn new_rename(id: Ident, to:Name,
                  tail: SyntaxContext) -> SyntaxContext {
    with_sctable(|table| new_rename_internal(id, to, tail, table))
}

// Extend a syntax context with a given rename and sctable
fn new_rename_internal(id: Ident,
                       to: Name,
                       tail: SyntaxContext,
                       table: &SCTable) -> SyntaxContext {
    let key = (tail,id,to);
    let mut rename_memo = table.rename_memo.borrow_mut();
    let new_ctxt = |_: &(SyntaxContext, Ident, Mrk)|
                   idx_push(table.table.borrow_mut().get(), Rename(id, to, tail));

    *rename_memo.get().find_or_insert_with(key, new_ctxt)
}

/// Fetch the SCTable from TLS, create one if it doesn't yet exist.
pub fn with_sctable<T>(op: |&SCTable| -> T) -> T {
    local_data_key!(sctable_key: Rc<SCTable>)

    local_data::get(sctable_key, |opt_ts| {
        let table = match opt_ts {
            None => {
                let ts = Rc::new(new_sctable_internal());
                local_data::set(sctable_key, ts.clone());
                ts
            }
            Some(ts) => ts.clone()
        };
        op(table.deref())
    })
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
    let table = table.table.borrow();
    for (idx,val) in table.get().iter().enumerate() {
        error!("{:4u} : {:?}",idx,val);
    }
}

/// Clear the tables from TLD to reclaim memory.
pub fn clear_tables() {
    with_sctable(|table| {
        *table.table.borrow_mut().get() = Vec::new();
        *table.mark_memo.borrow_mut().get() = HashMap::new();
        *table.rename_memo.borrow_mut().get() = HashMap::new();
    });
    with_resolve_table_mut(|table| *table = HashMap::new());
}

// Add a value to the end of a vec, return its index
fn idx_push<T>(vec: &mut Vec<T> , val: T) -> u32 {
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
fn with_resolve_table_mut<T>(op: |&mut ResolveTable| -> T) -> T {
    local_data_key!(resolve_table_key: Rc<RefCell<ResolveTable>>)

    local_data::get(resolve_table_key, |opt_ts| {
        let table = match opt_ts {
            None => {
                let ts = Rc::new(RefCell::new(HashMap::new()));
                local_data::set(resolve_table_key, ts.clone());
                ts
            }
            Some(ts) => ts.clone()
        };
        op(table.deref().borrow_mut().get())
    })
}

// Resolve a syntax object to a name, per MTWT.
// adding memorization to possibly resolve 500+ seconds in resolve for librustc (!)
fn resolve_internal(id: Ident,
                    table: &SCTable,
                    resolve_table: &mut ResolveTable) -> Name {
    let key = (id.name, id.ctxt);

    match resolve_table.find(&key) {
        Some(&name) => return name,
        None => {}
    }

    let resolved = {
        let result = *table.table.borrow().get().get(id.ctxt as uint);
        match result {
            EmptyCtxt => id.name,
            // ignore marks here:
            Mark(_,subctxt) =>
                resolve_internal(Ident{name:id.name, ctxt: subctxt},
                                 table, resolve_table),
            // do the rename if necessary:
            Rename(Ident{name, ctxt}, toname, subctxt) => {
                let resolvedfrom =
                    resolve_internal(Ident{name:name, ctxt:ctxt},
                                     table, resolve_table);
                let resolvedthis =
                    resolve_internal(Ident{name:id.name, ctxt:subctxt},
                                     table, resolve_table);
                if (resolvedthis == resolvedfrom)
                    && (marksof_internal(ctxt, resolvedthis, table)
                        == marksof_internal(subctxt, resolvedthis, table)) {
                    toname
                } else {
                    resolvedthis
                }
            }
            IllegalCtxt => fail!("expected resolvable context, got IllegalCtxt")
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
        let table_entry = {
            let table = table.table.borrow();
            *table.get().get(loopvar as uint)
        };
        match table_entry {
            EmptyCtxt => {
                return result;
            },
            Mark(mark, tl) => {
                xorPush(&mut result, mark);
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
            IllegalCtxt => fail!("expected resolvable context, got IllegalCtxt")
        }
    }
}

/// Return the outer mark for a context with a mark at the outside.
/// FAILS when outside is not a mark.
pub fn outer_mark(ctxt: SyntaxContext) -> Mrk {
    with_sctable(|sctable| {
        match *sctable.table.borrow().get().get(ctxt as uint) {
            Mark(mrk, _) => mrk,
            _ => fail!("can't retrieve outer mark when outside is not a mark")
        }
    })
}

// Push a name... unless it matches the one on top, in which
// case pop and discard (so two of the same marks cancel)
fn xorPush(marks: &mut Vec<Mrk>, mark: Mrk) {
    if (marks.len() > 0) && (*marks.last().unwrap() == mark) {
        marks.pop().unwrap();
    } else {
        marks.push(mark);
    }
}

#[cfg(test)]
mod tests {
    use ast::*;
    use super::{resolve, xorPush, new_mark_internal, new_sctable_internal};
    use super::{new_rename_internal, marksof_internal, resolve_internal};
    use super::{SCTable, EmptyCtxt, Mark, Rename, IllegalCtxt};
    use std::vec::Vec;
    use collections::HashMap;

    #[test] fn xorpush_test () {
        let mut s = Vec::new();
        xorPush(&mut s, 14);
        assert_eq!(s.clone(), vec!(14));
        xorPush(&mut s, 14);
        assert_eq!(s.clone(), Vec::new());
        xorPush(&mut s, 14);
        assert_eq!(s.clone(), vec!(14));
        xorPush(&mut s, 15);
        assert_eq!(s.clone(), vec!(14, 15));
        xorPush(&mut s, 16);
        assert_eq!(s.clone(), vec!(14, 15, 16));
        xorPush(&mut s, 16);
        assert_eq!(s.clone(), vec!(14, 15));
        xorPush(&mut s, 15);
        assert_eq!(s.clone(), vec!(14));
    }

    fn id(n: Name, s: SyntaxContext) -> Ident {
        Ident {name: n, ctxt: s}
    }

    // because of the SCTable, I now need a tidy way of
    // creating syntax objects. Sigh.
    #[deriving(Clone, Eq, Show)]
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
                      M(mrk) => new_mark_internal(mrk,tail,table),
                      R(ident,name) => new_rename_internal(ident,name,tail,table)}})
    }

    // gather a SyntaxContext back into a vector of TestSCs
    fn refold_test_sc(mut sc: SyntaxContext, table : &SCTable) -> Vec<TestSC> {
        let mut result = Vec::new();
        loop {
            let table = table.table.borrow();
            match *table.get().get(sc as uint) {
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
                IllegalCtxt => fail!("expected resolvable context, got IllegalCtxt")
            }
        }
    }

    #[test] fn test_unfold_refold(){
        let mut t = new_sctable_internal();

        let test_sc = vec!(M(3),R(id(101,0),14),M(9));
        assert_eq!(unfold_test_sc(test_sc.clone(),EMPTY_CTXT,&mut t),4);
        {
            let table = t.table.borrow();
            assert!(*table.get().get(2) == Mark(9,0));
            assert!(*table.get().get(3) == Rename(id(101,0),14,2));
            assert!(*table.get().get(4) == Mark(3,3));
        }
        assert_eq!(refold_test_sc(4,&t),test_sc);
    }

    // extend a syntax context with a sequence of marks given
    // in a vector. v[0] will be the outermost mark.
    fn unfold_marks(mrks: Vec<Mrk> , tail: SyntaxContext, table: &SCTable)
                    -> SyntaxContext {
        mrks.iter().rev().fold(tail, |tail:SyntaxContext, mrk:&Mrk|
                   {new_mark_internal(*mrk,tail,table)})
    }

    #[test] fn unfold_marks_test() {
        let mut t = new_sctable_internal();

        assert_eq!(unfold_marks(vec!(3,7),EMPTY_CTXT,&mut t),3);
        {
            let table = t.table.borrow();
            assert!(*table.get().get(2) == Mark(7,0));
            assert!(*table.get().get(3) == Mark(3,2));
        }
    }

    #[test] fn test_marksof () {
        let stopname = 242;
        let name1 = 243;
        let mut t = new_sctable_internal();
        assert_eq!(marksof_internal (EMPTY_CTXT,stopname,&t),Vec::new());
        // FIXME #5074: ANF'd to dodge nested calls
        { let ans = unfold_marks(vec!(4,98),EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans,stopname,&t),vec!(4,98));}
        // does xoring work?
        { let ans = unfold_marks(vec!(5,5,16),EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans,stopname,&t), vec!(16));}
        // does nested xoring work?
        { let ans = unfold_marks(vec!(5,10,10,5,16),EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans, stopname,&t), vec!(16));}
        // rename where stop doesn't match:
        { let chain = vec!(M(9),
                        R(id(name1,
                             new_mark_internal (4, EMPTY_CTXT,&mut t)),
                          100101102),
                        M(14));
         let ans = unfold_test_sc(chain,EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans, stopname, &t), vec!(9,14));}
        // rename where stop does match
        { let name1sc = new_mark_internal(4, EMPTY_CTXT, &mut t);
         let chain = vec!(M(9),
                       R(id(name1, name1sc),
                         stopname),
                       M(14));
         let ans = unfold_test_sc(chain,EMPTY_CTXT,&mut t);
         assert_eq! (marksof_internal (ans, stopname, &t), vec!(9)); }
    }


    #[test] fn resolve_tests () {
        let a = 40;
        let mut t = new_sctable_internal();
        let mut rt = HashMap::new();
        // - ctxt is MT
        assert_eq!(resolve_internal(id(a,EMPTY_CTXT),&mut t, &mut rt),a);
        // - simple ignored marks
        { let sc = unfold_marks(vec!(1,2,3),EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),a);}
        // - orthogonal rename where names don't match
        { let sc = unfold_test_sc(vec!(R(id(50,EMPTY_CTXT),51),M(12)),EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),a);}
        // - rename where names do match, but marks don't
        { let sc1 = new_mark_internal(1,EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(vec!(R(id(a,sc1),50),
                                   M(1),
                                   M(2)),
                                 EMPTY_CTXT,&mut t);
        assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), a);}
        // - rename where names and marks match
        { let sc1 = unfold_test_sc(vec!(M(1),M(2)),EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(vec!(R(id(a,sc1),50),M(1),M(2)),EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), 50); }
        // - rename where names and marks match by literal sharing
        { let sc1 = unfold_test_sc(vec!(M(1),M(2)),EMPTY_CTXT,&mut t);
         let sc = unfold_test_sc(vec!(R(id(a,sc1),50)),sc1,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), 50); }
        // - two renames of the same var.. can only happen if you use
        // local-expand to prevent the inner binding from being renamed
        // during the rename-pass caused by the first:
        println!("about to run bad test");
        { let sc = unfold_test_sc(vec!(R(id(a,EMPTY_CTXT),50),
                                    R(id(a,EMPTY_CTXT),51)),
                                  EMPTY_CTXT,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt), 51); }
        // the simplest double-rename:
        { let a_to_a50 = new_rename_internal(id(a,EMPTY_CTXT),50,EMPTY_CTXT,&mut t);
         let a50_to_a51 = new_rename_internal(id(a,a_to_a50),51,a_to_a50,&mut t);
         assert_eq!(resolve_internal(id(a,a50_to_a51),&mut t, &mut rt),51);
         // mark on the outside doesn't stop rename:
         let sc = new_mark_internal(9,a50_to_a51,&mut t);
         assert_eq!(resolve_internal(id(a,sc),&mut t, &mut rt),51);
         // but mark on the inside does:
         let a50_to_a51_b = unfold_test_sc(vec!(R(id(a,a_to_a50),51),
                                              M(9)),
                                           a_to_a50,
                                           &mut t);
         assert_eq!(resolve_internal(id(a,a50_to_a51_b),&mut t, &mut rt),50);}
    }

    #[test] fn mtwt_resolve_test(){
        let a = 40;
        assert_eq!(resolve(id(a,EMPTY_CTXT)),a);
    }


    #[test] fn hashing_tests () {
        let mut t = new_sctable_internal();
        assert_eq!(new_mark_internal(12,EMPTY_CTXT,&mut t),2);
        assert_eq!(new_mark_internal(13,EMPTY_CTXT,&mut t),3);
        // using the same one again should result in the same index:
        assert_eq!(new_mark_internal(12,EMPTY_CTXT,&mut t),2);
        // I'm assuming that the rename table will behave the same....
    }

    #[test] fn resolve_table_hashing_tests() {
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
}
