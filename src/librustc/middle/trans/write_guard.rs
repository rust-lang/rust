// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Logic relating to rooting and write guards for managed values.
//! This code is primarily for use by datum;
//! it exists in its own module both to keep datum.rs bite-sized
//! and for each in debugging (e.g., so you can use
//! `RUST_LOG=rustc::middle::trans::write_guard`).


use middle::borrowck::{RootInfo, root_map_key};
use middle::trans::cleanup;
use middle::trans::common::*;
use middle::trans::datum::*;
use syntax::codemap::Span;
use syntax::ast;

pub fn root_and_write_guard<'a, K:KindOps>(datum: &Datum<K>,
                                           bcx: &'a Block<'a>,
                                           span: Span,
                                           expr_id: ast::NodeId,
                                           derefs: uint) -> &'a Block<'a> {
    let key = root_map_key { id: expr_id, derefs: derefs };
    debug!("write_guard::root_and_write_guard(key={:?})", key);

    // root the autoderef'd value, if necessary:
    //
    // (Note: root'd values are always boxes)
    let ccx = bcx.ccx();
    let root_map = ccx.maps.root_map.borrow();
    match root_map.get().find(&key) {
        None => bcx,
        Some(&root_info) => root(datum, bcx, span, key, root_info)
    }
}

fn root<'a, K:KindOps>(datum: &Datum<K>,
                       bcx: &'a Block<'a>,
                       _span: Span,
                       root_key: root_map_key,
                       root_info: RootInfo) -> &'a Block<'a> {
    //! In some cases, borrowck will decide that an @T value must be
    //! rooted for the program to be safe.  In that case, we will call
    //! this function, which will stash a copy away until we exit the
    //! scope `scope_id`.

    debug!("write_guard::root(root_key={:?}, root_info={:?}, datum={:?})",
           root_key, root_info, datum.to_str(bcx.ccx()));

    // Root the datum. Note that we must zero this value,
    // because sometimes we root on one path but not another.
    // See e.g. #4904.
    lvalue_scratch_datum(
        bcx, datum.ty, "__write_guard", true,
        cleanup::AstScope(root_info.scope), (),
        |(), bcx, llval| datum.shallow_copy_and_take(bcx, llval)).bcx
}
