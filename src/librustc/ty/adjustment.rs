// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::{self, Ty, TyCtxt, TypeAndMut};
use ty::LvaluePreference::{NoPreference};

use syntax::ast;
use syntax_pos::Span;

use hir;

#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub struct Adjustment<'tcx> {
    pub kind: Adjust<'tcx>,
    pub target: Ty<'tcx>
}

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum Adjust<'tcx> {
    /// Go from ! to any type.
    NeverToAny,

    /// Go from a fn-item type to a fn-pointer type.
    ReifyFnPointer,

    /// Go from a safe fn pointer to an unsafe fn pointer.
    UnsafeFnPointer,

    /// Go from a mut raw pointer to a const raw pointer.
    MutToConstPointer,

    /// Represents coercing a pointer to a different kind of pointer - where 'kind'
    /// here means either or both of raw vs borrowed vs unique and fat vs thin.
    ///
    /// We transform pointers by following the following steps in order:
    /// 1. Deref the pointer `self.autoderefs` times (may be 0).
    /// 2. If `autoref` is `Some(_)`, then take the address and produce either a
    ///    `&` or `*` pointer.
    /// 3. If `unsize` is `Some(_)`, then apply the unsize transformation,
    ///    which will do things like convert thin pointers to fat
    ///    pointers, or convert structs containing thin pointers to
    ///    structs containing fat pointers, or convert between fat
    ///    pointers.  We don't store the details of how the transform is
    ///    done (in fact, we don't know that, because it might depend on
    ///    the precise type parameters). We just store the target
    ///    type. Trans figures out what has to be done at monomorphization
    ///    time based on the precise source/target type at hand.
    ///
    /// To make that more concrete, here are some common scenarios:
    ///
    /// 1. The simplest cases are where the pointer is not adjusted fat vs thin.
    /// Here the pointer will be dereferenced N times (where a dereference can
    /// happen to raw or borrowed pointers or any smart pointer which implements
    /// Deref, including Box<_>). The number of dereferences is given by
    /// `autoderefs`.  It can then be auto-referenced zero or one times, indicated
    /// by `autoref`, to either a raw or borrowed pointer. In these cases unsize is
    /// None.
    ///
    /// 2. A thin-to-fat coercon involves unsizing the underlying data. We start
    /// with a thin pointer, deref a number of times, unsize the underlying data,
    /// then autoref. The 'unsize' phase may change a fixed length array to a
    /// dynamically sized one, a concrete object to a trait object, or statically
    /// sized struct to a dyncamically sized one. E.g., &[i32; 4] -> &[i32] is
    /// represented by:
    ///
    /// ```
    /// Adjust::DerefRef {
    ///     autoderefs: 1,          // &[i32; 4] -> [i32; 4]
    ///     autoref: Some(AutoBorrow::Ref), // [i32] -> &[i32]
    ///     unsize: Some([i32]),    // [i32; 4] -> [i32]
    /// }
    /// ```
    ///
    /// Note that for a struct, the 'deep' unsizing of the struct is not recorded.
    /// E.g., `struct Foo<T> { x: T }` we can coerce &Foo<[i32; 4]> to &Foo<[i32]>
    /// The autoderef and -ref are the same as in the above example, but the type
    /// stored in `unsize` is `Foo<[i32]>`, we don't store any further detail about
    /// the underlying conversions from `[i32; 4]` to `[i32]`.
    ///
    /// 3. Coercing a `Box<T>` to `Box<Trait>` is an interesting special case.  In
    /// that case, we have the pointer we need coming in, so there are no
    /// autoderefs, and no autoref. Instead we just do the `Unsize` transformation.
    /// At some point, of course, `Box` should move out of the compiler, in which
    /// case this is analogous to transformating a struct. E.g., Box<[i32; 4]> ->
    /// Box<[i32]> is represented by:
    ///
    /// ```
    /// Adjust::DerefRef {
    ///     autoderefs: 0,
    ///     autoref: None,
    ///     unsize: Some(Box<[i32]>),
    /// }
    /// ```
    DerefRef {
        /// Step 1. Apply a number of dereferences, producing an lvalue.
        autoderefs: usize,

        /// Step 2. Optionally produce a pointer/reference from the value.
        autoref: Option<AutoBorrow<'tcx>>,

        /// Step 3. Unsize a pointer/reference value, e.g. `&[T; n]` to
        /// `&[T]`. Note that the source could be a thin or fat pointer.
        unsize: bool,
    }
}

impl<'tcx> Adjustment<'tcx> {
    pub fn is_identity(&self) -> bool {
        match self.kind {
            Adjust::NeverToAny => self.target.is_never(),

            Adjust::DerefRef { autoderefs: 0, autoref: None, unsize: false } => true,

            Adjust::ReifyFnPointer |
            Adjust::UnsafeFnPointer |
            Adjust::MutToConstPointer |
            Adjust::DerefRef {..} => false,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, RustcEncodable, RustcDecodable)]
pub enum AutoBorrow<'tcx> {
    /// Convert from T to &T.
    Ref(&'tcx ty::Region, hir::Mutability),

    /// Convert from T to *T.
    RawPtr(hir::Mutability),
}

#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug)]
pub enum CustomCoerceUnsized {
    /// Records the index of the field being coerced.
    Struct(usize)
}

impl<'a, 'gcx, 'tcx> ty::TyS<'tcx> {
    pub fn adjust_for_autoderef<F>(&'tcx self,
                                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                   expr_id: ast::NodeId,
                                   expr_span: Span,
                                   autoderef: u32, // how many autoderefs so far?
                                   mut method_type: F)
                                   -> Ty<'tcx> where
        F: FnMut(ty::MethodCall) -> Option<Ty<'tcx>>,
    {
        let method_call = ty::MethodCall::autoderef(expr_id, autoderef);
        let mut adjusted_ty = self;
        if let Some(method_ty) = method_type(method_call) {
            // Method calls always have all late-bound regions
            // fully instantiated.
            adjusted_ty = tcx.no_late_bound_regions(&method_ty.fn_ret()).unwrap();
        }
        match adjusted_ty.builtin_deref(true, NoPreference) {
            Some(mt) => mt.ty,
            None => {
                span_bug!(
                    expr_span,
                    "the {}th autoderef for {} failed: {}",
                    autoderef,
                    expr_id,
                    adjusted_ty);
            }
        }
    }

    pub fn adjust_for_autoref(&'tcx self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                              autoref: Option<AutoBorrow<'tcx>>)
                              -> Ty<'tcx> {
        match autoref {
            None => self,
            Some(AutoBorrow::Ref(r, m)) => {
                tcx.mk_ref(r, TypeAndMut { ty: self, mutbl: m })
            }
            Some(AutoBorrow::RawPtr(m)) => {
                tcx.mk_ptr(TypeAndMut { ty: self, mutbl: m })
            }
        }
    }
}
