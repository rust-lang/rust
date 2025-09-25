//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] check-pass

// Make sure we support non-call operations for opaque types even if
// its not part of its item bounds.

use std::ops::{Deref, DerefMut, Index, IndexMut};

fn mk<T>() -> T {
    todo!()
}

fn add() -> impl Sized {
    let unconstrained = if false {
        add() + 1
        //[current]~^ ERROR cannot add `{integer}` to `impl Sized
    } else {
        let with_infer = mk();
        let _ = with_infer + 1;
        with_infer
    };
    let _: u32 = unconstrained;
    1u32
}

fn mul_assign() -> impl Sized {
    if false {
        let mut temp = mul_assign();
        temp *= 2;
        //[current]~^ ERROR binary assignment operation `*=` cannot be applied to type `impl Sized`
    }

    let mut with_infer = mk();
    with_infer *= 2;
    let _: u32 = with_infer;

    1u32
}

struct DerefWrapper<T>(T);
impl<T: Deref> Deref for DerefWrapper<T> {
    type Target = T::Target;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}
impl<T: DerefMut> DerefMut for DerefWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

fn explicit_deref() -> DerefWrapper<impl Sized> {
    if false {
        let _rarw = &*explicit_deref();
        //[current]~^ ERROR type `DerefWrapper<impl Sized>` cannot be dereferenced

        let mut with_infer = DerefWrapper(mk());
        let _rarw = &*with_infer;
        with_infer
    } else {
        DerefWrapper(&1u32)
    }
}
fn explicit_deref_mut() -> DerefWrapper<impl Sized> {
    if false {
        *explicit_deref_mut() = 1;
        //[current]~^ ERROR type `DerefWrapper<impl Sized>` cannot be dereferenced

        let mut with_infer = DerefWrapper(Default::default());
        *with_infer = 1;
        with_infer
    } else {
        DerefWrapper(Box::new(1u32))
    }
}

struct IndexWrapper<T>(T);
impl<T: Index<U>, U> Index<U> for IndexWrapper<T> {
    type Output = T::Output;
    fn index(&self, index: U) -> &Self::Output {
        &self.0[index]
    }
}
impl<T: IndexMut<U>, U> IndexMut<U> for IndexWrapper<T> {
    fn index_mut(&mut self, index: U) -> &mut Self::Output {
        &mut self.0[index]
    }
}
fn explicit_index() -> IndexWrapper<impl Sized> {
    if false {
        let _y = explicit_index()[0];
        //[current]~^ ERROR the type `impl Sized` cannot be indexed by `_`

        let with_infer = IndexWrapper(Default::default());
        let _y = with_infer[0];
        with_infer
    } else {
        IndexWrapper([1u32])
    }
}
fn explicit_index_mut() -> IndexWrapper<impl Sized> {
    if false {
        explicit_index_mut()[0] = 1;
        //[current]~^ ERROR the type `impl Sized` cannot be indexed by `_`

        let mut with_infer = IndexWrapper(Default::default());
        with_infer[0] = 1;
        with_infer
    } else {
        IndexWrapper([1u32])
    }
}

fn main() {}
