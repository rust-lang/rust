//! An iterator over the type substructure.
//! WARNING: this does not keep track of the region depth.

use smallvec::{SmallVec, smallvec};
use tracing::debug;

use crate::data_structures::SsoHashSet;
use crate::inherent::*;
use crate::{self as ty, Interner};

// The TypeWalker's stack is hot enough that it's worth going to some effort to
// avoid heap allocations.
type TypeWalkerStack<I> = SmallVec<[<I as Interner>::GenericArg; 8]>;

/// An iterator for walking the type tree.
///
/// It's very easy to produce a deeply
/// nested type tree with a lot of
/// identical subtrees. In order to work efficiently
/// in this situation walker only visits each type once.
/// It maintains a set of visited types and
/// skips any types that are already there.
pub struct TypeWalker<I: Interner> {
    stack: TypeWalkerStack<I>,
    last_subtree: usize,
    pub visited: SsoHashSet<I::GenericArg>,
}

impl<I: Interner> TypeWalker<I> {
    pub fn new(root: I::GenericArg) -> Self {
        Self { stack: smallvec![root], last_subtree: 1, visited: SsoHashSet::new() }
    }

    /// Skips the subtree corresponding to the last type
    /// returned by `next()`.
    ///
    /// Example: Imagine you are walking `Foo<Bar<i32>, usize>`.
    ///
    /// ```ignore (illustrative)
    /// let mut iter: TypeWalker = ...;
    /// iter.next(); // yields Foo
    /// iter.next(); // yields Bar<i32>
    /// iter.skip_current_subtree(); // skips i32
    /// iter.next(); // yields usize
    /// ```
    pub fn skip_current_subtree(&mut self) {
        self.stack.truncate(self.last_subtree);
    }
}

impl<I: Interner> Iterator for TypeWalker<I> {
    type Item = I::GenericArg;

    fn next(&mut self) -> Option<I::GenericArg> {
        debug!("next(): stack={:?}", self.stack);
        loop {
            let next = self.stack.pop()?;
            self.last_subtree = self.stack.len();
            if self.visited.insert(next) {
                push_inner::<I>(&mut self.stack, next);
                debug!("next: stack={:?}", self.stack);
                return Some(next);
            }
        }
    }
}

/// We push `GenericArg`s on the stack in reverse order so as to
/// maintain a pre-order traversal. As of the time of this
/// writing, the fact that the traversal is pre-order is not
/// known to be significant to any code, but it seems like the
/// natural order one would expect (basically, the order of the
/// types as they are written).
fn push_inner<I: Interner>(stack: &mut TypeWalkerStack<I>, parent: I::GenericArg) {
    match parent.kind() {
        ty::GenericArgKind::Type(parent_ty) => match parent_ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Infer(_)
            | ty::Param(_)
            | ty::Never
            | ty::Error(_)
            | ty::Placeholder(..)
            | ty::Bound(..)
            | ty::Foreign(..) => {}

            ty::Pat(ty, pat) => {
                push_ty_pat::<I>(stack, pat);
                stack.push(ty.into());
            }
            ty::Array(ty, len) => {
                stack.push(len.into());
                stack.push(ty.into());
            }
            ty::Slice(ty) => {
                stack.push(ty.into());
            }
            ty::RawPtr(ty, _) => {
                stack.push(ty.into());
            }
            ty::Ref(lt, ty, _) => {
                stack.push(ty.into());
                stack.push(lt.into());
            }
            ty::Alias(_, data) => {
                stack.extend(data.args.iter().rev());
            }
            ty::Dynamic(obj, lt) => {
                stack.push(lt.into());
                stack.extend(
                    obj.iter()
                        .rev()
                        .filter_map(|predicate| {
                            let (args, opt_ty) = match predicate.skip_binder() {
                                ty::ExistentialPredicate::Trait(tr) => (tr.args, None),
                                ty::ExistentialPredicate::Projection(p) => (p.args, Some(p.term)),
                                ty::ExistentialPredicate::AutoTrait(_) => {
                                    return None;
                                }
                            };

                            Some(args.iter().rev().chain(opt_ty.map(|term| match term.kind() {
                                ty::TermKind::Ty(ty) => ty.into(),
                                ty::TermKind::Const(ct) => ct.into(),
                            })))
                        })
                        .flatten(),
                );
            }
            ty::Adt(_, args)
            | ty::Closure(_, args)
            | ty::CoroutineClosure(_, args)
            | ty::Coroutine(_, args)
            | ty::CoroutineWitness(_, args)
            | ty::FnDef(_, args) => {
                stack.extend(args.iter().rev());
            }
            ty::Tuple(ts) => stack.extend(ts.iter().rev().map(|ty| ty.into())),
            ty::FnPtr(sig_tys, _hdr) => {
                stack.extend(
                    sig_tys.skip_binder().inputs_and_output.iter().rev().map(|ty| ty.into()),
                );
            }
            ty::UnsafeBinder(bound_ty) => {
                stack.push(bound_ty.skip_binder().into());
            }
        },
        ty::GenericArgKind::Lifetime(_) => {}
        ty::GenericArgKind::Const(parent_ct) => match parent_ct.kind() {
            ty::ConstKind::Infer(_)
            | ty::ConstKind::Param(_)
            | ty::ConstKind::Placeholder(_)
            | ty::ConstKind::Bound(..)
            | ty::ConstKind::Error(_) => {}

            ty::ConstKind::Value(cv) => stack.push(cv.ty().into()),

            ty::ConstKind::Expr(expr) => stack.extend(expr.args().iter().rev()),
            ty::ConstKind::Unevaluated(ct) => {
                stack.extend(ct.args.iter().rev());
            }
        },
    }
}

fn push_ty_pat<I: Interner>(stack: &mut TypeWalkerStack<I>, pat: I::Pat) {
    match pat.kind() {
        ty::PatternKind::Range { start, end } => {
            stack.push(end.into());
            stack.push(start.into());
        }
        ty::PatternKind::Or(pats) => {
            for pat in pats.iter() {
                push_ty_pat::<I>(stack, pat)
            }
        }
    }
}
