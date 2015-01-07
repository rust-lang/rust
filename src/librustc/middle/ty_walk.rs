// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An iterator over the type substructure.

use middle::ty::{self, Ty};
use std::iter::Iterator;

pub struct TypeWalker<'tcx> {
    stack: Vec<Ty<'tcx>>,
    last_subtree: uint,
}

impl<'tcx> TypeWalker<'tcx> {
    pub fn new(ty: Ty<'tcx>) -> TypeWalker<'tcx> {
        TypeWalker { stack: vec!(ty), last_subtree: 1, }
    }

    fn push_subtypes(&mut self, parent_ty: Ty<'tcx>) {
        match parent_ty.sty {
            ty::ty_bool | ty::ty_char | ty::ty_int(_) | ty::ty_uint(_) | ty::ty_float(_) |
            ty::ty_str | ty::ty_infer(_) | ty::ty_param(_) | ty::ty_err => {
            }
            ty::ty_uniq(ty) | ty::ty_vec(ty, _) | ty::ty_open(ty) => {
                self.stack.push(ty);
            }
            ty::ty_ptr(ref mt) | ty::ty_rptr(_, ref mt) => {
                self.stack.push(mt.ty);
            }
            ty::ty_projection(ref data) => {
                self.push_reversed(data.trait_ref.substs.types.as_slice());
            }
            ty::ty_trait(box ty::TyTrait { ref principal, .. }) => {
                self.push_reversed(principal.substs().types.as_slice());
            }
            ty::ty_enum(_, ref substs) |
            ty::ty_struct(_, ref substs) |
            ty::ty_unboxed_closure(_, _, ref substs) => {
                self.push_reversed(substs.types.as_slice());
            }
            ty::ty_tup(ref ts) => {
                self.push_reversed(ts.as_slice());
            }
            ty::ty_bare_fn(_, ref ft) => {
                self.push_sig_subtypes(&ft.sig);
            }
        }
    }

    fn push_sig_subtypes(&mut self, sig: &ty::PolyFnSig<'tcx>) {
        match sig.0.output {
            ty::FnConverging(output) => { self.stack.push(output); }
            ty::FnDiverging => { }
        }
        self.push_reversed(sig.0.inputs.as_slice());
    }

    fn push_reversed(&mut self, tys: &[Ty<'tcx>]) {
        // We push slices on the stack in reverse order so as to
        // maintain a pre-order traversal. As of the time of this
        // writing, the fact that the traversal is pre-order is not
        // known to be significant to any code, but it seems like the
        // natural order one would expect (basically, the order of the
        // types as they are written).
        for &ty in tys.iter().rev() {
            self.stack.push(ty);
        }
    }

    /// Skips the subtree of types corresponding to the last type
    /// returned by `next()`.
    ///
    /// Example: Imagine you are walking `Foo<Bar<int>, uint>`.
    ///
    /// ```rust
    /// let mut iter: TypeWalker = ...;
    /// iter.next(); // yields Foo
    /// iter.next(); // yields Bar<int>
    /// iter.skip_current_subtree(); // skips int
    /// iter.next(); // yields uint
    /// ```
    pub fn skip_current_subtree(&mut self) {
        self.stack.truncate(self.last_subtree);
    }
}

impl<'tcx> Iterator for TypeWalker<'tcx> {
    type Item = Ty<'tcx>;

    fn next(&mut self) -> Option<Ty<'tcx>> {
        debug!("next(): stack={:?}", self.stack);
        match self.stack.pop() {
            None => {
                return None;
            }
            Some(ty) => {
                self.last_subtree = self.stack.len();
                self.push_subtypes(ty);
                debug!("next: stack={:?}", self.stack);
                Some(ty)
            }
        }
    }
}
