// Regression test for #74429, where we didn't think that a type parameter
// outlived `ReEmpty`.

//@ check-pass

use std::marker::PhantomData;
use std::ptr::NonNull;

pub unsafe trait RawData {
    type Elem;
}

unsafe impl<A> RawData for OwnedRepr<A> {
    type Elem = A;
}

unsafe impl<'a, A> RawData for ViewRepr<&'a A> {
    type Elem = A;
}

pub struct OwnedRepr<A> {
    ptr: PhantomData<A>,
}

// these Copy impls are not necessary for the repro, but allow the code to compile without error
// on 1.44.1
#[derive(Copy, Clone)]
pub struct ViewRepr<A> {
    life: PhantomData<A>,
}

#[derive(Copy, Clone)]
pub struct ArrayBase<S>
where
    S: RawData,
{
    ptr: NonNull<S::Elem>,
}

pub type Array<A> = ArrayBase<OwnedRepr<A>>;

pub type ArrayView<'a, A> = ArrayBase<ViewRepr<&'a A>>;

impl<A, S> ArrayBase<S>
where
    S: RawData<Elem = A>,
{
    pub fn index_axis(&self) -> ArrayView<'_, A> {
        unimplemented!()
    }

    pub fn axis_iter<'a>(&'a self) -> std::iter::Empty<&'a A> {
        unimplemented!()
    }
}

pub fn x<T: Copy>(a: Array<T>) {
    // _ just avoids a must_use warning
    let _ = (0..1).filter(|_| true);
    let y = a.index_axis();
    a.axis_iter().for_each(|_| {
        let _ = y;
    });
}

fn main() {}
