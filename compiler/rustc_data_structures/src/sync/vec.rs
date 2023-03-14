use std::marker::PhantomData;

use rustc_index::vec::Idx;

pub struct AppendOnlyIndexVec<I: Idx, T: Copy> {
    #[cfg(not(parallel_compiler))]
    vec: elsa::vec::FrozenVec<T>,
    #[cfg(parallel_compiler)]
    vec: elsa::sync::LockFreeFrozenVec<T>,
    _marker: PhantomData<fn(&I)>,
}

impl<I: Idx, T: Copy> AppendOnlyIndexVec<I, T> {
    pub fn new() -> Self {
        Self {
            #[cfg(not(parallel_compiler))]
            vec: elsa::vec::FrozenVec::new(),
            #[cfg(parallel_compiler)]
            vec: elsa::sync::LockFreeFrozenVec::new(),
            _marker: PhantomData,
        }
    }

    pub fn push(&self, val: T) -> I {
        #[cfg(not(parallel_compiler))]
        let i = self.vec.len();
        #[cfg(not(parallel_compiler))]
        self.vec.push(val);
        #[cfg(parallel_compiler)]
        let i = self.vec.push(val);
        I::new(i)
    }

    pub fn get(&self, i: I) -> Option<T> {
        let i = i.index();
        #[cfg(not(parallel_compiler))]
        return self.vec.get_copy(i);
        #[cfg(parallel_compiler)]
        return self.vec.get(i);
    }
}

pub struct AppendOnlyVec<T: Copy> {
    #[cfg(not(parallel_compiler))]
    vec: elsa::vec::FrozenVec<T>,
    #[cfg(parallel_compiler)]
    vec: elsa::sync::LockFreeFrozenVec<T>,
}

impl<T: Copy> AppendOnlyVec<T> {
    pub fn new() -> Self {
        Self {
            #[cfg(not(parallel_compiler))]
            vec: elsa::vec::FrozenVec::new(),
            #[cfg(parallel_compiler)]
            vec: elsa::sync::LockFreeFrozenVec::new(),
        }
    }

    pub fn push(&self, val: T) {
        #[cfg(not(parallel_compiler))]
        self.vec.push(val);
        #[cfg(parallel_compiler)]
        self.vec.push(val)
    }

    pub fn get(&self, i: usize) -> Option<T> {
        #[cfg(not(parallel_compiler))]
        return self.vec.get_copy(i);
        #[cfg(parallel_compiler)]
        return self.vec.get(i);
    }
}
