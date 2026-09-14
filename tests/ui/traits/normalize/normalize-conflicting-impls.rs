fn problematic_function<Space>(material_surface_element: ())
where
    DefaultAllocator: FiniteElementAllocator<(), Space>,
{
    let _: Point2<f64> = material_surface_element.map_reference_coords().into();
}

impl<N, R> Allocator<N, R> for DefaultAllocator
where
    R::Value: DimName, //~ ERROR: `Value` not found for `R`
{
    type Buffer = ();
}
impl<N> Allocator<N, ()> for DefaultAllocator {}
//~^ ERROR: conflicting implementations
impl DimName for () {}
impl DimName for u32 {}
impl<N, D: DimName> From<VectorN<N, D>> for Point<N, D> {
    fn from(_: VectorN<N, D>) -> Self {
        todo!()
    }
}

impl FiniteElement<u32> for () {}

type VectorN<N, D> = Matrix<<DefaultAllocator as Allocator<N, D>>::Buffer>;

type Point2<N> = Point<N, u32>;

struct DefaultAllocator;
struct Matrix<S>(S);
struct Point<N, D>(N, D);

trait Allocator<Scalar, R> {
    type Buffer;
}
trait DimName {}
trait FiniteElementAllocator<GeometryDim, NodalDim>:
    Allocator<f64, ()> + Allocator<f64, NodalDim>
{
}

trait FiniteElement<GeometryDim> {
    fn map_reference_coords(&self) -> VectorN<f64, GeometryDim> {
        todo!()
    }
}

fn main() {}
