// Regression test for #66768.
//@ check-pass
#![allow(dead_code)]
//-^ "dead code" is needed to reproduce the issue.

use std::marker::PhantomData;
use std::ops::{Add, Mul};

fn problematic_function<Space>(material_surface_element: Edge2dElement)
where
    DefaultAllocator: FiniteElementAllocator<DimU1, Space>,
{
    let _: Point2<f64> = material_surface_element.map_reference_coords().into();
}

impl<T> ArrayLength<T> for UTerm {
    type ArrayType = ();
}
impl<T, N: ArrayLength<T>> ArrayLength<T> for UInt<N, B0> {
    type ArrayType = GenericArrayImplEven<T, N>;
}
impl<T, N: ArrayLength<T>> ArrayLength<T> for UInt<N, B1> {
    type ArrayType = GenericArrayImplOdd<T, N>;
}
impl<U> Add<U> for UTerm {
    type Output = U;
    fn add(self, _: U) -> Self::Output {
        unimplemented!()
    }
}
impl<Ul, Ur> Add<UInt<Ur, B1>> for UInt<Ul, B0>
where
    Ul: Add<Ur>,
{
    type Output = UInt<Sum<Ul, Ur>, B1>;
    fn add(self, _: UInt<Ur, B1>) -> Self::Output {
        unimplemented!()
    }
}
impl<U> Mul<U> for UTerm {
    type Output = UTerm;
    fn mul(self, _: U) -> Self {
        unimplemented!()
    }
}
impl<Ul, B, Ur> Mul<UInt<Ur, B>> for UInt<Ul, B0>
where
    Ul: Mul<UInt<Ur, B>>,
{
    type Output = UInt<Prod<Ul, UInt<Ur, B>>, B0>;
    fn mul(self, _: UInt<Ur, B>) -> Self::Output {
        unimplemented!()
    }
}
impl<Ul, B, Ur> Mul<UInt<Ur, B>> for UInt<Ul, B1>
where
    Ul: Mul<UInt<Ur, B>>,
    UInt<Prod<Ul, UInt<Ur, B>>, B0>: Add<UInt<Ur, B>>,
{
    type Output = Sum<UInt<Prod<Ul, UInt<Ur, B>>, B0>, UInt<Ur, B>>;
    fn mul(self, _: UInt<Ur, B>) -> Self::Output {
        unimplemented!()
    }
}
impl<N, R, C> Allocator<N, R, C> for DefaultAllocator
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    type Buffer = ArrayStorage<N, R, C>;
    fn allocate_uninitialized(_: R, _: C) -> Self::Buffer {
        unimplemented!()
    }
    fn allocate_from_iterator<I>(_: R, _: C, _: I) -> Self::Buffer {
        unimplemented!()
    }
}
impl<N, C> Allocator<N, Dynamic, C> for DefaultAllocator {
    type Buffer = VecStorage<N, Dynamic, C>;
    fn allocate_uninitialized(_: Dynamic, _: C) -> Self::Buffer {
        unimplemented!()
    }
    fn allocate_from_iterator<I>(_: Dynamic, _: C, _: I) -> Self::Buffer {
        unimplemented!()
    }
}
impl DimName for DimU1 {
    type Value = U1;
    fn name() -> Self {
        unimplemented!()
    }
}
impl DimName for DimU2 {
    type Value = U2;
    fn name() -> Self {
        unimplemented!()
    }
}
impl<N, D> From<VectorN<N, D>> for Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    fn from(_: VectorN<N, D>) -> Self {
        unimplemented!()
    }
}
impl<GeometryDim, NodalDim> FiniteElementAllocator<GeometryDim, NodalDim> for DefaultAllocator where
    DefaultAllocator: Allocator<f64, GeometryDim> + Allocator<f64, NodalDim>
{
}
impl ReferenceFiniteElement for Edge2dElement {
    type NodalDim = DimU1;
}
impl FiniteElement<DimU2> for Edge2dElement {
    fn map_reference_coords(&self) -> Vector2<f64> {
        unimplemented!()
    }
}

type Owned<N, R, C> = <DefaultAllocator as Allocator<N, R, C>>::Buffer;
type MatrixMN<N, R, C> = Matrix<N, R, C, Owned<N, R, C>>;
type VectorN<N, D> = MatrixMN<N, D, DimU1>;
type Vector2<N> = VectorN<N, DimU2>;
type Point2<N> = Point<N, DimU2>;
type U1 = UInt<UTerm, B1>;
type U2 = UInt<UInt<UTerm, B1>, B0>;
type Sum<A, B> = <A as Add<B>>::Output;
type Prod<A, B> = <A as Mul<B>>::Output;

struct GenericArray<T, U: ArrayLength<T>> {
    _data: U::ArrayType,
}
struct GenericArrayImplEven<T, U> {
    _parent2: U,
    _marker: T,
}
struct GenericArrayImplOdd<T, U> {
    _parent2: U,
    _data: T,
}
struct B0;
struct B1;
struct UTerm;
struct UInt<U, B> {
    _marker: PhantomData<(U, B)>,
}
struct DefaultAllocator;
struct Dynamic;
struct DimU1;
struct DimU2;
struct Matrix<N, R, C, S> {
    _data: S,
    _phantoms: PhantomData<(N, R, C)>,
}
struct ArrayStorage<N, R, C>
where
    R: DimName,
    C: DimName,
    R::Value: Mul<C::Value>,
    Prod<R::Value, C::Value>: ArrayLength<N>,
{
    _data: GenericArray<N, Prod<R::Value, C::Value>>,
}
struct VecStorage<N, R, C> {
    _data: N,
    _nrows: R,
    _ncols: C,
}
struct Point<N, D>
where
    DefaultAllocator: Allocator<N, D>,
{
    _coords: VectorN<N, D>,
}
struct Edge2dElement;

trait ArrayLength<T> {
    type ArrayType;
}
trait Allocator<Scalar, R, C = DimU1> {
    type Buffer;
    fn allocate_uninitialized(nrows: R, ncols: C) -> Self::Buffer;
    fn allocate_from_iterator<I>(nrows: R, ncols: C, iter: I) -> Self::Buffer;
}
trait DimName {
    type Value;
    fn name() -> Self;
}
trait FiniteElementAllocator<GeometryDim, NodalDim>:
    Allocator<f64, GeometryDim> + Allocator<f64, NodalDim>
{
}
trait ReferenceFiniteElement {
    type NodalDim;
}
trait FiniteElement<GeometryDim>: ReferenceFiniteElement
where
    DefaultAllocator: FiniteElementAllocator<GeometryDim, Self::NodalDim>,
{
    fn map_reference_coords(&self) -> VectorN<f64, GeometryDim>;
}

fn main() {}
