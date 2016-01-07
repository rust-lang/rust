// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::AutoAdjustment::*;
pub use self::AutoRef::*;

use middle::ty::{self, Ty, TypeAndMut, TypeFoldable};
use middle::ty::LvaluePreference::{NoPreference};

use syntax::ast;
use syntax::codemap::Span;

use rustc_front::hir;

#[derive(Copy, Clone)]
pub enum AutoAdjustment<'tcx> {
    AdjustReifyFnPointer,   // go from a fn-item type to a fn-pointer type
    AdjustUnsafeFnPointer,  // go from a safe fn pointer to an unsafe fn pointer
    AdjustDerefRef(AutoDerefRef<'tcx>),
}

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
/// AutoDerefRef {
///     autoderefs: 1,          // &[i32; 4] -> [i32; 4]
///     autoref: Some(AutoPtr), // [i32] -> &[i32]
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
/// AutoDerefRef {
///     autoderefs: 0,
///     autoref: None,
///     unsize: Some(Box<[i32]>),
/// }
/// ```
#[derive(Copy, Clone)]
pub struct AutoDerefRef<'tcx> {
    /// Step 1. Apply a number of dereferences, producing an lvalue.
    pub autoderefs: usize,

    /// Step 2. Optionally produce a pointer/reference from the value.
    pub autoref: Option<AutoRef<'tcx>>,

    /// Step 3. Unsize a pointer/reference value, e.g. `&[T; n]` to
    /// `&[T]`. The stored type is the target pointer type. Note that
    /// the source could be a thin or fat pointer.
    pub unsize: Option<Ty<'tcx>>,
}

impl<'tcx> AutoAdjustment<'tcx> {
    pub fn is_identity(&self) -> bool {
        match *self {
            AdjustReifyFnPointer |
            AdjustUnsafeFnPointer => false,
            AdjustDerefRef(ref r) => r.is_identity(),
        }
    }
}
impl<'tcx> AutoDerefRef<'tcx> {
    pub fn is_identity(&self) -> bool {
        self.autoderefs == 0 && self.unsize.is_none() && self.autoref.is_none()
    }
}


#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AutoRef<'tcx> {
    /// Convert from T to &T.
    AutoPtr(&'tcx ty::Region, hir::Mutability),

    /// Convert from T to *T.
    /// Value to thin pointer.
    AutoUnsafe(hir::Mutability),
}

#[derive(Clone, Copy, RustcEncodable, RustcDecodable, Debug)]
pub enum CustomCoerceUnsized {
    /// Records the index of the field being coerced.
    Struct(usize)
}

impl<'tcx> ty::TyS<'tcx> {
    /// See `expr_ty_adjusted`
    pub fn adjust<F>(&'tcx self, cx: &ty::ctxt<'tcx>,
                     span: Span,
                     expr_id: ast::NodeId,
                     adjustment: Option<&AutoAdjustment<'tcx>>,
                     mut method_type: F)
                     -> Ty<'tcx> where
        F: FnMut(ty::MethodCall) -> Option<Ty<'tcx>>,
    {
        if let ty::TyError = self.sty {
            return self;
        }

        return match adjustment {
            Some(adjustment) => {
                match *adjustment {
                   AdjustReifyFnPointer => {
                        match self.sty {
                            ty::TyBareFn(Some(_), b) => {
                                cx.mk_fn(None, b)
                            }
                            _ => {
                                cx.sess.bug(
                                    &format!("AdjustReifyFnPointer adjustment on non-fn-item: \
                                              {:?}", self));
                            }
                        }
                    }

                   AdjustUnsafeFnPointer => {
                        match self.sty {
                            ty::TyBareFn(None, b) => cx.safe_to_unsafe_fn_ty(b),
                            ref b => {
                                cx.sess.bug(
                                    &format!("AdjustReifyFnPointer adjustment on non-fn-item: \
                                             {:?}",
                                            b));
                            }
                        }
                   }

                    AdjustDerefRef(ref adj) => {
                        let mut adjusted_ty = self;

                        if !adjusted_ty.references_error() {
                            for i in 0..adj.autoderefs {
                                adjusted_ty =
                                    adjusted_ty.adjust_for_autoderef(cx,
                                                                     expr_id,
                                                                     span,
                                                                     i as u32,
                                                                     &mut method_type);
                            }
                        }

                        if let Some(target) = adj.unsize {
                            target
                        } else {
                            adjusted_ty.adjust_for_autoref(cx, adj.autoref)
                        }
                    }
                }
            }
            None => self
        };
    }

    pub fn adjust_for_autoderef<F>(&'tcx self,
                                   cx: &ty::ctxt<'tcx>,
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
            let fn_ret = cx.no_late_bound_regions(&method_ty.fn_ret()).unwrap();
            adjusted_ty = fn_ret.unwrap();
        }
        match adjusted_ty.builtin_deref(true, NoPreference) {
            Some(mt) => mt.ty,
            None => {
                cx.sess.span_bug(
                    expr_span,
                    &format!("the {}th autoderef failed: {}",
                             autoderef,
                             adjusted_ty)
                        );
            }
        }
    }

    pub fn adjust_for_autoref(&'tcx self, cx: &ty::ctxt<'tcx>,
                              autoref: Option<AutoRef<'tcx>>)
                              -> Ty<'tcx> {
        match autoref {
            None => self,
            Some(AutoPtr(r, m)) => {
                cx.mk_ref(r, TypeAndMut { ty: self, mutbl: m })
            }
            Some(AutoUnsafe(m)) => {
                cx.mk_ptr(TypeAndMut { ty: self, mutbl: m })
            }
        }
    }
}
