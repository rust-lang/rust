//! An iterator over the type substructure.
//! WARNING: this does not keep track of the region depth.

use crate::ty::{self, Ty};
use smallvec::{self, SmallVec};
use crate::mir::interpret::ConstValue;

// The TypeWalker's stack is hot enough that it's worth going to some effort to
// avoid heap allocations.
pub type TypeWalkerArray<'tcx> = [Ty<'tcx>; 8];
pub type TypeWalkerStack<'tcx> = SmallVec<TypeWalkerArray<'tcx>>;

pub struct TypeWalker<'tcx> {
    stack: TypeWalkerStack<'tcx>,
    last_subtree: usize,
}

impl<'tcx> TypeWalker<'tcx> {
    pub fn new(ty: Ty<'tcx>) -> TypeWalker<'tcx> {
        TypeWalker { stack: smallvec![ty], last_subtree: 1, }
    }

    /// Skips the subtree of types corresponding to the last type
    /// returned by `next()`.
    ///
    /// Example: Imagine you are walking `Foo<Bar<int>, usize>`.
    ///
    /// ```
    /// let mut iter: TypeWalker = ...;
    /// iter.next(); // yields Foo
    /// iter.next(); // yields Bar<int>
    /// iter.skip_current_subtree(); // skips int
    /// iter.next(); // yields usize
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
                None
            }
            Some(ty) => {
                self.last_subtree = self.stack.len();
                push_subtypes(&mut self.stack, ty);
                debug!("next: stack={:?}", self.stack);
                Some(ty)
            }
        }
    }
}

pub fn walk_shallow(ty: Ty<'_>) -> smallvec::IntoIter<TypeWalkerArray<'_>> {
    let mut stack = SmallVec::new();
    push_subtypes(&mut stack, ty);
    stack.into_iter()
}

// We push types on the stack in reverse order so as to
// maintain a pre-order traversal. As of the time of this
// writing, the fact that the traversal is pre-order is not
// known to be significant to any code, but it seems like the
// natural order one would expect (basically, the order of the
// types as they are written).
fn push_subtypes<'tcx>(stack: &mut TypeWalkerStack<'tcx>, parent_ty: Ty<'tcx>) {
    match parent_ty.sty {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) |
        ty::Str | ty::Infer(_) | ty::Param(_) | ty::Never | ty::Error |
        ty::Placeholder(..) | ty::Bound(..) | ty::Foreign(..) => {
        }
        ty::Array(ty, len) => {
            if let ConstValue::Unevaluated(_, substs) = len.val {
                stack.extend(substs.types().rev());
            }
            stack.push(len.ty);
            stack.push(ty);
        }
        ty::Slice(ty) => {
            stack.push(ty);
        }
        ty::RawPtr(ref mt) => {
            stack.push(mt.ty);
        }
        ty::Ref(_, ty, _) => {
            stack.push(ty);
        }
        ty::Projection(ref data) | ty::UnnormalizedProjection(ref data) => {
            stack.extend(data.substs.types().rev());
        }
        ty::Dynamic(ref obj, ..) => {
            stack.extend(obj.iter().rev().flat_map(|predicate| {
                let (substs, opt_ty) = match *predicate.skip_binder() {
                    ty::ExistentialPredicate::Trait(tr) => (tr.substs, None),
                    ty::ExistentialPredicate::Projection(p) =>
                        (p.substs, Some(p.ty)),
                    ty::ExistentialPredicate::AutoTrait(_) =>
                        // Empty iterator
                        (ty::InternalSubsts::empty(), None),
                };

                substs.types().rev().chain(opt_ty)
            }));
        }
        ty::Adt(_, substs) | ty::Opaque(_, substs) => {
            stack.extend(substs.types().rev());
        }
        ty::Closure(_, ref substs) => {
            stack.extend(substs.substs.types().rev());
        }
        ty::Generator(_, ref substs, _) => {
            stack.extend(substs.substs.types().rev());
        }
        ty::GeneratorWitness(ts) => {
            stack.extend(ts.skip_binder().iter().cloned().rev());
        }
        ty::Tuple(ts) => {
            stack.extend(ts.iter().map(|k| k.expect_ty()).rev());
        }
        ty::FnDef(_, substs) => {
            stack.extend(substs.types().rev());
        }
        ty::FnPtr(sig) => {
            stack.push(sig.skip_binder().output());
            stack.extend(sig.skip_binder().inputs().iter().cloned().rev());
        }
    }
}
