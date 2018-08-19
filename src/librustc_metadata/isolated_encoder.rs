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
use rustc::ty::TyCtxt;
use rustc_serialize::Encodable;

/// The IsolatedEncoder provides facilities to write to crate metadata while
/// making sure that anything going through it is also feed into an ICH hasher.
pub struct IsolatedEncoder<'a, 'b: 'a, 'tcx: 'b> {
    pub tcx: TyCtxt<'b, 'tcx, 'tcx>,
    ecx: &'a mut EncodeContext<'b, 'tcx>,
}

impl<'a, 'b: 'a, 'tcx: 'b> IsolatedEncoder<'a, 'b, 'tcx> {

    pub fn new(ecx: &'a mut EncodeContext<'b, 'tcx>) -> Self {
        let tcx = ecx.tcx;
        IsolatedEncoder {
            tcx,
            ecx,
        }
    }

    pub fn lazy<T>(&mut self, value: &T) -> Lazy<T>
        where T: Encodable
    {
        self.ecx.lazy(value)
    }

    pub fn lazy_seq<I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = T>,
              T: Encodable
    {
        self.ecx.lazy_seq(iter)
    }

    pub fn lazy_seq_ref<'x, I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = &'x T>,
              T: 'x + Encodable
    {
        self.ecx.lazy_seq_ref(iter)
    }

    pub fn lazy_seq_from_slice<T>(&mut self, slice: &[T]) -> LazySeq<T>
        where T: Encodable
    {
        self.ecx.lazy_seq_ref(slice.iter())
    }
}
