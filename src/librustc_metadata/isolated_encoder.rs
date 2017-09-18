// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use encoder::EncodeContext;
use schema::{Lazy, LazySeq};

use rustc::ich::{StableHashingContext, Fingerprint};
use rustc::ty::TyCtxt;

use rustc_data_structures::accumulate_vec::AccumulateVec;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable};
use rustc_serialize::Encodable;

/// The IsolatedEncoder provides facilities to write to crate metadata while
/// making sure that anything going through it is also feed into an ICH hasher.
pub struct IsolatedEncoder<'a, 'b: 'a, 'tcx: 'b> {
    pub tcx: TyCtxt<'b, 'tcx, 'tcx>,
    ecx: &'a mut EncodeContext<'b, 'tcx>,
    hcx: Option<(StableHashingContext<'tcx>, StableHasher<Fingerprint>)>,
}

impl<'a, 'b: 'a, 'tcx: 'b> IsolatedEncoder<'a, 'b, 'tcx> {

    pub fn new(ecx: &'a mut EncodeContext<'b, 'tcx>) -> Self {
        let tcx = ecx.tcx;
        let compute_ich = ecx.compute_ich;
        IsolatedEncoder {
            tcx,
            ecx,
            hcx: if compute_ich {
                // We are always hashing spans for things in metadata because
                // don't know if a downstream crate will use them or not.
                // Except when -Zquery-dep-graph is specified because we don't
                // want to mess up our tests.
                let hcx = if tcx.sess.opts.debugging_opts.query_dep_graph {
                    tcx.create_stable_hashing_context()
                } else {
                    tcx.create_stable_hashing_context().force_span_hashing()
                };

                Some((hcx, StableHasher::new()))
            } else {
                None
            }
        }
    }

    pub fn finish(self) -> (Option<Fingerprint>, &'a mut EncodeContext<'b, 'tcx>) {
        if let Some((_, hasher)) = self.hcx {
            (Some(hasher.finish()), self.ecx)
        } else {
            (None, self.ecx)
        }
    }

    pub fn lazy<T>(&mut self, value: &T) -> Lazy<T>
        where T: Encodable + HashStable<StableHashingContext<'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            value.hash_stable(hcx, hasher);
            debug!("metadata-hash: {:?}", hasher);
        }
        self.ecx.lazy(value)
    }

    pub fn lazy_seq<I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = T>,
              T: Encodable + HashStable<StableHashingContext<'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            let iter = iter.into_iter();
            let (lower_bound, upper_bound) = iter.size_hint();

            if upper_bound == Some(lower_bound) {
                lower_bound.hash_stable(hcx, hasher);
                let mut num_items_hashed = 0;
                let ret = self.ecx.lazy_seq(iter.inspect(|item| {
                    item.hash_stable(hcx, hasher);
                    num_items_hashed += 1;
                }));

                // Sometimes items in a sequence are filtered out without being
                // hashed (e.g. for &[ast::Attribute]) and this code path cannot
                // handle that correctly, so we want to make sure we didn't hit
                // it by accident.
                if lower_bound != num_items_hashed {
                    bug!("Hashed a different number of items ({}) than expected ({})",
                         num_items_hashed,
                         lower_bound);
                }
                debug!("metadata-hash: {:?}", hasher);
                ret
            } else {
                // Collect into a vec so we know the length of the sequence
                let items: AccumulateVec<[T; 32]> = iter.collect();
                items.hash_stable(hcx, hasher);
                debug!("metadata-hash: {:?}", hasher);
                self.ecx.lazy_seq(items)
            }
        } else {
            self.ecx.lazy_seq(iter)
        }
    }

    pub fn lazy_seq_ref<'x, I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = &'x T>,
              T: 'x + Encodable + HashStable<StableHashingContext<'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            let iter = iter.into_iter();
            let (lower_bound, upper_bound) = iter.size_hint();

            if upper_bound == Some(lower_bound) {
                lower_bound.hash_stable(hcx, hasher);
                let mut num_items_hashed = 0;
                let ret = self.ecx.lazy_seq_ref(iter.inspect(|item| {
                    item.hash_stable(hcx, hasher);
                    num_items_hashed += 1;
                }));

                // Sometimes items in a sequence are filtered out without being
                // hashed (e.g. for &[ast::Attribute]) and this code path cannot
                // handle that correctly, so we want to make sure we didn't hit
                // it by accident.
                if lower_bound != num_items_hashed {
                    bug!("Hashed a different number of items ({}) than expected ({})",
                         num_items_hashed,
                         lower_bound);
                }
                debug!("metadata-hash: {:?}", hasher);
                ret
            } else {
                // Collect into a vec so we know the length of the sequence
                let items: AccumulateVec<[&'x T; 32]> = iter.collect();
                items.hash_stable(hcx, hasher);
                debug!("metadata-hash: {:?}", hasher);
                self.ecx.lazy_seq_ref(items.iter().map(|x| *x))
            }
        } else {
            self.ecx.lazy_seq_ref(iter)
        }
    }

    pub fn lazy_seq_from_slice<T>(&mut self, slice: &[T]) -> LazySeq<T>
        where T: Encodable + HashStable<StableHashingContext<'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            slice.hash_stable(hcx, hasher);
            debug!("metadata-hash: {:?}", hasher);
        }
        self.ecx.lazy_seq_ref(slice.iter())
    }

    pub fn lazy_seq_ref_from_slice<T>(&mut self, slice: &[&T]) -> LazySeq<T>
        where T: Encodable + HashStable<StableHashingContext<'tcx>>
    {
        if let Some((ref mut hcx, ref mut hasher)) = self.hcx {
            slice.hash_stable(hcx, hasher);
            debug!("metadata-hash: {:?}", hasher);
        }
        self.ecx.lazy_seq_ref(slice.iter().map(|x| *x))
    }
}
