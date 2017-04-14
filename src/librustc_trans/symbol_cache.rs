// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::rc::Rc;
use rustc::ty::TyCtxt;
use trans_item::TransItem;
use util::nodemap::FxHashMap;

// In the SymbolCache we collect the symbol names of translation items
// and cache them for later reference. This is just a performance
// optimization and the cache is populated lazilly; symbol names of
// translation items are deterministic and fully defined by the item.
// Thus they can always be recomputed if needed.

pub struct SymbolCache<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    index: RefCell<FxHashMap<TransItem<'tcx>, Rc<String>>>,
}

impl<'a, 'tcx> SymbolCache<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        SymbolCache {
            tcx: tcx,
            index: RefCell::new(FxHashMap())
        }
    }

    pub fn get(&self, trans_item: TransItem<'tcx>) -> Rc<String> {
        let mut index = self.index.borrow_mut();
        index.entry(trans_item)
             .or_insert_with(|| Rc::new(trans_item.compute_symbol_name(self.tcx)))
             .clone()
    }
}
