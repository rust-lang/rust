//@ check-pass
//@ compile-flags: -C debug_assertions=yes -Zunstable-options

#[allow(dead_code)]
fn problematic_function<Space>()
where
    DefaultAlloc: FinAllok<R1, Space>,
{
    let e = Edge2dElement;
    let _ = Into::<Point>::into(e.map_reference_coords());
}
impl<N> Allocator<N, R0> for DefaultAlloc {
    type Buffer = MStorage;
}
impl<N> Allocator<N, R1> for DefaultAlloc {
    type Buffer = MStorage;
}
impl<N, D> From<VectorN<N, D>> for Point
where
    DefaultAlloc: Allocator<N, D>,
{
    fn from(_: VectorN<N, D>) -> Self {
        unimplemented!()
    }
}
impl<GeometryDim, NodalDim> FinAllok<GeometryDim, NodalDim> for DefaultAlloc
where
    DefaultAlloc: Allocator<Ure, GeometryDim>,
    DefaultAlloc: Allocator<Ure, NodalDim>
{
}
impl FiniteElement<R1> for Edge2dElement {
    fn map_reference_coords(&self) -> VectorN<Ure, R1> {
        unimplemented!()
    }
}
type VectorN<N, R> = (N, R, <DefaultAlloc as Allocator<N, R>>::Buffer);
struct DefaultAlloc;
struct R0;
struct R1;
struct MStorage;
struct Point;
struct Edge2dElement;
struct Ure;
trait Allocator<N, R> {
    type Buffer;
}
trait FinAllok<GeometryDim, NodalDim>:
    Allocator<Ure, GeometryDim> +
    Allocator<Ure, NodalDim> +
{
}
trait FiniteElement<Rau>
where
    DefaultAlloc: FinAllok<Rau, Rau>,
{
    fn map_reference_coords(&self) -> VectorN<Ure, Rau>;
}
fn main() {}
