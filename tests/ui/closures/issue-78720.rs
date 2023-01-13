fn server() -> impl {
//~^ ERROR at least one trait must be specified
    ().map2(|| "")
}

trait FilterBase2 {
    fn map2<F>(self, f: F) -> Map2<F> {}
    //~^ ERROR mismatched types
    //~^^ ERROR the size for values of type `Self` cannot be known at compilation time
}

struct Map2<Segment2> {
    _func: F,
    //~^ ERROR cannot find type `F` in this scope
}

impl<F> FilterBase2 for F {}

fn main() {}
