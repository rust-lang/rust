// Regression test for #61311
// We would ICE after failing to normalize `Self::Proj` in the `impl` below.

// build-pass (FIXME(62277): could be check-pass?)

pub struct Unit;
trait Obj {}

trait Bound {}
impl Bound for Unit {}

pub trait HasProj {
    type Proj;
}

impl<T> HasProj for T {
    type Proj = Unit;
}

trait HasProjFn {
    type Proj;
    fn the_fn(_: Self::Proj);
}

impl HasProjFn for Unit
where
    Box<dyn Obj + 'static>: HasProj,
    <Box<dyn Obj + 'static> as HasProj>::Proj: Bound,
{
    type Proj = Unit;
    fn the_fn(_: Self::Proj) {}
}

fn main() {}
