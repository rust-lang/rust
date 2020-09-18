//! An iterator over the type substructure.
//! WARNING: this does not keep track of the region depth.

use crate::ty;
use crate::ty::subst::{GenericArg, GenericArgKind};
use arrayvec::ArrayVec;
use rustc_data_structures::fx::FxHashSet;
use smallvec::{self, SmallVec};
use std::hash::Hash;

/// Small-storage-optimized implementation of a set
/// made specifically for walking type tree.
///
/// Stores elements in a small array up to a certain length
/// and switches to `HashSet` when that length is exceeded.
pub enum MiniSet<T> {
    Array(ArrayVec<[T; 8]>),
    Set(FxHashSet<T>),
}

impl<T: Eq + Hash + Copy> MiniSet<T> {
    /// Creates an empty `MiniSet`.
    pub fn new() -> Self {
        MiniSet::Array(ArrayVec::new())
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, true is returned.
    ///
    /// If the set did have this value present, false is returned.
    pub fn insert(&mut self, elem: T) -> bool {
        match self {
            MiniSet::Array(array) => {
                if array.iter().any(|e| *e == elem) {
                    false
                } else {
                    if array.try_push(elem).is_err() {
                        let mut set: FxHashSet<T> = array.iter().copied().collect();
                        set.insert(elem);
                        *self = MiniSet::Set(set);
                    }
                    true
                }
            }
            MiniSet::Set(set) => set.insert(elem),
        }
    }
}

// The TypeWalker's stack is hot enough that it's worth going to some effort to
// avoid heap allocations.
type TypeWalkerStack<'tcx> = SmallVec<[GenericArg<'tcx>; 8]>;

pub struct TypeWalker<'tcx> {
    stack: TypeWalkerStack<'tcx>,
    last_subtree: usize,
    visited: MiniSet<GenericArg<'tcx>>,
}

/// An iterator for walking the type tree.
///
/// It's very easy to produce a deeply
/// nested type tree with a lot of
/// identical subtrees. In order to work efficiently
/// in this situation walker only visits each type once.
/// It maintains a set of visited types and
/// skips any types that are already there.
impl<'tcx> TypeWalker<'tcx> {
    pub fn new(root: GenericArg<'tcx>) -> Self {
        Self { stack: smallvec![root], last_subtree: 1, visited: MiniSet::new() }
    }

    /// Skips the subtree corresponding to the last type
    /// returned by `next()`.
    ///
    /// Example: Imagine you are walking `Foo<Bar<i32>, usize>`.
    ///
    /// ```
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

impl<'tcx> Iterator for TypeWalker<'tcx> {
    type Item = GenericArg<'tcx>;

    fn next(&mut self) -> Option<GenericArg<'tcx>> {
        debug!("next(): stack={:?}", self.stack);
        loop {
            let next = self.stack.pop()?;
            self.last_subtree = self.stack.len();
            if self.visited.insert(next) {
                push_inner(&mut self.stack, next);
                debug!("next: stack={:?}", self.stack);
                return Some(next);
            }
        }
    }
}

impl GenericArg<'tcx> {
    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(self) -> TypeWalker<'tcx> {
        TypeWalker::new(self)
    }

    /// Iterator that walks the immediate children of `self`. Hence
    /// `Foo<Bar<i32>, u32>` yields the sequence `[Bar<i32>, u32]`
    /// (but not `i32`, like `walk`).
    ///
    /// Iterator only walks items once.
    /// It accepts visited set, updates it with all visited types
    /// and skips any types that are already there.
    pub fn walk_shallow(
        self,
        visited: &mut MiniSet<GenericArg<'tcx>>,
    ) -> impl Iterator<Item = GenericArg<'tcx>> {
        let mut stack = SmallVec::new();
        push_inner(&mut stack, self);
        stack.retain(|a| visited.insert(*a));
        stack.into_iter()
    }
}

impl<'tcx> super::TyS<'tcx> {
    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(&'tcx self) -> TypeWalker<'tcx> {
        TypeWalker::new(self.into())
    }
}

// We push `GenericArg`s on the stack in reverse order so as to
// maintain a pre-order traversal. As of the time of this
// writing, the fact that the traversal is pre-order is not
// known to be significant to any code, but it seems like the
// natural order one would expect (basically, the order of the
// types as they are written).
fn push_inner<'tcx>(stack: &mut TypeWalkerStack<'tcx>, parent: GenericArg<'tcx>) {
    match parent.unpack() {
        GenericArgKind::Type(parent_ty) => match *parent_ty.kind() {
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

            ty::Array(ty, len) => {
                stack.push(len.into());
                stack.push(ty.into());
            }
            ty::Slice(ty) => {
                stack.push(ty.into());
            }
            ty::RawPtr(mt) => {
                stack.push(mt.ty.into());
            }
            ty::Ref(lt, ty, _) => {
                stack.push(ty.into());
                stack.push(lt.into());
            }
            ty::Projection(data) => {
                stack.extend(data.substs.iter().rev());
            }
            ty::Dynamic(obj, lt) => {
                stack.push(lt.into());
                stack.extend(obj.iter().rev().flat_map(|predicate| {
                    let (substs, opt_ty) = match predicate.skip_binder() {
                        ty::ExistentialPredicate::Trait(tr) => (tr.substs, None),
                        ty::ExistentialPredicate::Projection(p) => (p.substs, Some(p.ty)),
                        ty::ExistentialPredicate::AutoTrait(_) =>
                        // Empty iterator
                        {
                            (ty::InternalSubsts::empty(), None)
                        }
                    };

                    substs.iter().rev().chain(opt_ty.map(|ty| ty.into()))
                }));
            }
            ty::Adt(_, substs)
            | ty::Opaque(_, substs)
            | ty::Closure(_, substs)
            | ty::Generator(_, substs, _)
            | ty::Tuple(substs)
            | ty::FnDef(_, substs) => {
                stack.extend(substs.iter().rev());
            }
            ty::GeneratorWitness(ts) => {
                stack.extend(ts.skip_binder().iter().rev().map(|ty| ty.into()));
            }
            ty::FnPtr(sig) => {
                stack.push(sig.skip_binder().output().into());
                stack.extend(sig.skip_binder().inputs().iter().copied().rev().map(|ty| ty.into()));
            }
        },
        GenericArgKind::Lifetime(_) => {}
        GenericArgKind::Const(parent_ct) => {
            stack.push(parent_ct.ty.into());
            match parent_ct.val {
                ty::ConstKind::Infer(_)
                | ty::ConstKind::Param(_)
                | ty::ConstKind::Placeholder(_)
                | ty::ConstKind::Bound(..)
                | ty::ConstKind::Value(_)
                | ty::ConstKind::Error(_) => {}

                ty::ConstKind::Unevaluated(_, substs, _) => {
                    stack.extend(substs.iter().rev());
                }
            }
        }
    }
}
