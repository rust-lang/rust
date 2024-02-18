//@ compile-flags: --crate-type lib
//@ check-pass
//
// Regression test for issue #84399
// Tests that we keep the full `ParamEnv` when
// caching predicates with freshened types in the global cache

use std::marker::PhantomData;
pub trait Allocator<R> {
    type Buffer;
}
pub struct DefaultAllocator;
impl <R> Allocator<R> for DefaultAllocator {
    type Buffer = ();
}
pub type Owned<R> = <DefaultAllocator as Allocator<R>>::Buffer;
pub type MatrixMN<R> = Matrix<R, Owned<R>>;
pub type Matrix4<N> = Matrix<N, ()>;
pub struct Matrix<R, S> {
    pub data: S,
    _phantoms: PhantomData<R>,
}
pub fn set_object_transform(matrix: &Matrix4<()>) {
    matrix.js_buffer_view();
}
pub trait Storable {
    type Cell;
    fn slice_to_items(_buffer: &()) -> &[Self::Cell] {
        unimplemented!()
    }
}
pub type Cell<T> = <T as Storable>::Cell;
impl<R> Storable for MatrixMN<R>
where
    DefaultAllocator: Allocator<R>,
{
    type Cell = ();
}
pub trait JsBufferView {
    fn js_buffer_view(&self) -> usize {
        unimplemented!()
    }
}
impl<R> JsBufferView for [MatrixMN<R>]
where
    DefaultAllocator: Allocator<R>,
    MatrixMN<R>: Storable,
    [Cell<MatrixMN<R>>]: JsBufferView,
{
    fn js_buffer_view(&self) -> usize {
        <MatrixMN<R> as Storable>::slice_to_items(&()).js_buffer_view()
    }
}
impl JsBufferView for [()] {}
impl<R> JsBufferView for MatrixMN<R> where DefaultAllocator: Allocator<R> {}
