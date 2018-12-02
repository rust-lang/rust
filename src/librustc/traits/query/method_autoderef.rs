// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::sync::Lrc;
use infer::canonical::{Canonical, QueryResponse};
use ty::Ty;

#[derive(Debug)]
pub struct CandidateStep<'tcx> {
    pub self_ty: Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
    pub autoderefs: usize,
    // true if the type results from a dereference of a raw pointer.
    // when assembling candidates, we include these steps, but not when
    // picking methods. This so that if we have `foo: *const Foo` and `Foo` has methods
    // `fn by_raw_ptr(self: *const Self)` and `fn by_ref(&self)`, then
    // `foo.by_raw_ptr()` will work and `foo.by_ref()` won't.
    pub from_unsafe_deref: bool,
    pub unsize: bool,
}

#[derive(Clone, Debug)]
pub struct MethodAutoderefStepsResult<'tcx> {
    /// The valid autoderef steps that could be find.
    pub steps: Lrc<Vec<CandidateStep<'tcx>>>,
    /// If Some(T), a type autoderef reported an error on.
    pub opt_bad_ty: Option<Lrc<MethodAutoderefBadTy<'tcx>>>,
    /// If `true`, `steps` has been truncated due to reaching the
    /// recursion limit.
    pub reached_recursion_limit: bool,
}

#[derive(Debug)]
pub struct MethodAutoderefBadTy<'tcx> {
    pub reached_raw_pointer: bool,
    pub ty: Canonical<'tcx, QueryResponse<'tcx, Ty<'tcx>>>,
}

impl_stable_hash_for!(struct MethodAutoderefBadTy<'tcx> {
    reached_raw_pointer, ty
});

impl_stable_hash_for!(struct MethodAutoderefStepsResult<'tcx> {
    reached_recursion_limit, steps, opt_bad_ty
});

impl_stable_hash_for!(struct CandidateStep<'tcx> {
    self_ty, autoderefs, from_unsafe_deref, unsize
});
