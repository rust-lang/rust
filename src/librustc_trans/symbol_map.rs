// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use context::SharedCrateContext;
use monomorphize::Instance;
use rustc::ty::TyCtxt;
use std::borrow::Cow;
use syntax::codemap::Span;
use trans_item::TransItem;
use util::nodemap::FxHashMap;

// In the SymbolMap we collect the symbol names of all translation items of
// the current crate. This map exists as a performance optimization. Symbol
// names of translation items are deterministic and fully defined by the item.
// Thus they could also always be recomputed if needed.

pub struct SymbolMap<'tcx> {
    index: FxHashMap<TransItem<'tcx>, (usize, usize)>,
    arena: String,
}

impl<'tcx> SymbolMap<'tcx> {

    pub fn build<'a, I>(scx: &SharedCrateContext<'a, 'tcx>,
                        trans_items: I)
                        -> SymbolMap<'tcx>
        where I: Iterator<Item=TransItem<'tcx>>
    {
        // Check for duplicate symbol names
        let mut symbols: Vec<_> = trans_items.map(|trans_item| {
            (trans_item, trans_item.compute_symbol_name(scx))
        }).collect();

        (&mut symbols[..]).sort_by(|&(_, ref sym1), &(_, ref sym2)|{
            sym1.cmp(sym2)
        });

        for pair in (&symbols[..]).windows(2) {
            let sym1 = &pair[0].1;
            let sym2 = &pair[1].1;

            if *sym1 == *sym2 {
                let trans_item1 = pair[0].0;
                let trans_item2 = pair[1].0;

                let span1 = get_span(scx.tcx(), trans_item1);
                let span2 = get_span(scx.tcx(), trans_item2);

                // Deterministically select one of the spans for error reporting
                let span = match (span1, span2) {
                    (Some(span1), Some(span2)) => {
                        Some(if span1.lo.0 > span2.lo.0 {
                            span1
                        } else {
                            span2
                        })
                    }
                    (Some(span), None) |
                    (None, Some(span)) => Some(span),
                    _ => None
                };

                let error_message = format!("symbol `{}` is already defined", sym1);

                if let Some(span) = span {
                    scx.sess().span_fatal(span, &error_message)
                } else {
                    scx.sess().fatal(&error_message)
                }
            }
        }

        let mut symbol_map = SymbolMap {
            index: FxHashMap(),
            arena: String::with_capacity(1024),
        };

        for (trans_item, symbol) in symbols {
            let start_index = symbol_map.arena.len();
            symbol_map.arena.push_str(&symbol[..]);
            let end_index = symbol_map.arena.len();
            let prev_entry = symbol_map.index.insert(trans_item,
                                                     (start_index, end_index));
            if prev_entry.is_some() {
                bug!("TransItem encountered twice?")
            }
        }

        fn get_span<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              trans_item: TransItem<'tcx>) -> Option<Span> {
            match trans_item {
                TransItem::Fn(Instance { def, .. }) => {
                    tcx.map.as_local_node_id(def)
                }
                TransItem::Static(node_id) => Some(node_id),
                TransItem::DropGlue(_) => None,
            }.map(|node_id| {
                tcx.map.span(node_id)
            })
        }

        symbol_map
    }

    pub fn get(&self, trans_item: TransItem<'tcx>) -> Option<&str> {
        self.index.get(&trans_item).map(|&(start_index, end_index)| {
            &self.arena[start_index .. end_index]
        })
    }

    pub fn get_or_compute<'map, 'scx>(&'map self,
                                      scx: &SharedCrateContext<'scx, 'tcx>,
                                      trans_item: TransItem<'tcx>)
                                      -> Cow<'map, str> {
        if let Some(sym) = self.get(trans_item) {
            Cow::from(sym)
        } else {
            Cow::from(trans_item.compute_symbol_name(scx))
        }
    }
}
