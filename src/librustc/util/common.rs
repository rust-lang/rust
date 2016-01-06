// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use std::cell::{RefCell, Cell};
use std::collections::HashMap;
use std::collections::hash_state::HashState;
use std::ffi::CString;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::repeat;
use std::path::Path;
use std::time::Instant;

use rustc_front::hir;
use rustc_front::intravisit;
use rustc_front::intravisit::Visitor;

// The name of the associated type for `Fn` return types
pub const FN_OUTPUT_NAME: &'static str = "Output";

// Useful type to use with `Result<>` indicate that an error has already
// been reported to the user, so no need to continue checking.
#[derive(Clone, Copy, Debug)]
pub struct ErrorReported;

pub fn time<T, F>(do_it: bool, what: &str, f: F) -> T where
    F: FnOnce() -> T,
{
    thread_local!(static DEPTH: Cell<usize> = Cell::new(0));
    if !do_it { return f(); }

    let old = DEPTH.with(|slot| {
        let r = slot.get();
        slot.set(r + 1);
        r
    });

    let start = Instant::now();
    let rv = f();
    let dur = start.elapsed();

    // Hack up our own formatting for the duration to make it easier for scripts
    // to parse (always use the same number of decimal places and the same unit).
    const NANOS_PER_SEC: f64 = 1_000_000_000.0;
    let secs = dur.as_secs() as f64;
    let secs = secs + dur.subsec_nanos() as f64 / NANOS_PER_SEC;

    let mem_string = match get_resident() {
        Some(n) => {
            let mb = n as f64 / 1_000_000.0;
            format!("; rss: {}MB", mb.round() as usize)
        }
        None => "".to_owned(),
    };
    println!("{}time: {:.3}{}\t{}", repeat("  ").take(old).collect::<String>(),
             secs, mem_string, what);

    DEPTH.with(|slot| slot.set(old));

    rv
}

// Like std::macros::try!, but for Option<>.
macro_rules! option_try(
    ($e:expr) => (match $e { Some(e) => e, None => return None })
);

// Memory reporting
#[cfg(unix)]
fn get_resident() -> Option<usize> {
    use std::fs::File;
    use std::io::Read;

    let field = 1;
    let mut f = option_try!(File::open("/proc/self/statm").ok());
    let mut contents = String::new();
    option_try!(f.read_to_string(&mut contents).ok());
    let s = option_try!(contents.split_whitespace().nth(field));
    let npages = option_try!(s.parse::<usize>().ok());
    Some(npages * 4096)
}

#[cfg(windows)]
fn get_resident() -> Option<usize> {
    type BOOL = i32;
    type DWORD = u32;
    type HANDLE = *mut u8;
    use libc::size_t;
    use std::mem;
    #[repr(C)] #[allow(non_snake_case)]
    struct PROCESS_MEMORY_COUNTERS {
        cb: DWORD,
        PageFaultCount: DWORD,
        PeakWorkingSetSize: size_t,
        WorkingSetSize: size_t,
        QuotaPeakPagedPoolUsage: size_t,
        QuotaPagedPoolUsage: size_t,
        QuotaPeakNonPagedPoolUsage: size_t,
        QuotaNonPagedPoolUsage: size_t,
        PagefileUsage: size_t,
        PeakPagefileUsage: size_t,
    }
    type PPROCESS_MEMORY_COUNTERS = *mut PROCESS_MEMORY_COUNTERS;
    #[link(name = "psapi")]
    extern "system" {
        fn GetCurrentProcess() -> HANDLE;
        fn GetProcessMemoryInfo(Process: HANDLE,
                                ppsmemCounters: PPROCESS_MEMORY_COUNTERS,
                                cb: DWORD) -> BOOL;
    }
    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { mem::zeroed() };
    pmc.cb = mem::size_of_val(&pmc) as DWORD;
    match unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) } {
        0 => None,
        _ => Some(pmc.WorkingSetSize as usize),
    }
}

pub fn indent<R, F>(op: F) -> R where
    R: Debug,
    F: FnOnce() -> R,
{
    // Use in conjunction with the log post-processor like `src/etc/indenter`
    // to make debug output more readable.
    debug!(">>");
    let r = op();
    debug!("<< (Result = {:?})", r);
    r
}

pub struct Indenter {
    _cannot_construct_outside_of_this_module: ()
}

impl Drop for Indenter {
    fn drop(&mut self) { debug!("<<"); }
}

pub fn indenter() -> Indenter {
    debug!(">>");
    Indenter { _cannot_construct_outside_of_this_module: () }
}

struct LoopQueryVisitor<P> where P: FnMut(&hir::Expr_) -> bool {
    p: P,
    flag: bool,
}

impl<'v, P> Visitor<'v> for LoopQueryVisitor<P> where P: FnMut(&hir::Expr_) -> bool {
    fn visit_expr(&mut self, e: &hir::Expr) {
        self.flag |= (self.p)(&e.node);
        match e.node {
          // Skip inner loops, since a break in the inner loop isn't a
          // break inside the outer loop
          hir::ExprLoop(..) | hir::ExprWhile(..) => {}
          _ => intravisit::walk_expr(self, e)
        }
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn loop_query<P>(b: &hir::Block, p: P) -> bool where P: FnMut(&hir::Expr_) -> bool {
    let mut v = LoopQueryVisitor {
        p: p,
        flag: false,
    };
    intravisit::walk_block(&mut v, b);
    return v.flag;
}

struct BlockQueryVisitor<P> where P: FnMut(&hir::Expr) -> bool {
    p: P,
    flag: bool,
}

impl<'v, P> Visitor<'v> for BlockQueryVisitor<P> where P: FnMut(&hir::Expr) -> bool {
    fn visit_expr(&mut self, e: &hir::Expr) {
        self.flag |= (self.p)(e);
        intravisit::walk_expr(self, e)
    }
}

// Takes a predicate p, returns true iff p is true for any subexpressions
// of b -- skipping any inner loops (loop, while, loop_body)
pub fn block_query<P>(b: &hir::Block, p: P) -> bool where P: FnMut(&hir::Expr) -> bool {
    let mut v = BlockQueryVisitor {
        p: p,
        flag: false,
    };
    intravisit::walk_block(&mut v, &*b);
    return v.flag;
}

pub trait MemoizationMap {
    type Key: Clone;
    type Value: Clone;

    /// If `key` is present in the map, return the valuee,
    /// otherwise invoke `op` and store the value in the map.
    ///
    /// NB: if the receiver is a `DepTrackingMap`, special care is
    /// needed in the `op` to ensure that the correct edges are
    /// added into the dep graph. See the `DepTrackingMap` impl for
    /// more details!
    fn memoize<OP>(&self, key: Self::Key, op: OP) -> Self::Value
        where OP: FnOnce() -> Self::Value;
}

impl<K, V, S> MemoizationMap for RefCell<HashMap<K,V,S>>
    where K: Hash+Eq+Clone, V: Clone, S: HashState
{
    type Key = K;
    type Value = V;

    fn memoize<OP>(&self, key: K, op: OP) -> V
        where OP: FnOnce() -> V
    {
        let result = self.borrow().get(&key).cloned();
        match result {
            Some(result) => result,
            None => {
                let result = op();
                self.borrow_mut().insert(key, result.clone());
                result
            }
        }
    }
}

#[cfg(unix)]
pub fn path2cstr(p: &Path) -> CString {
    use std::os::unix::prelude::*;
    use std::ffi::OsStr;
    let p: &OsStr = p.as_ref();
    CString::new(p.as_bytes()).unwrap()
}
#[cfg(windows)]
pub fn path2cstr(p: &Path) -> CString {
    CString::new(p.to_str().unwrap()).unwrap()
}
