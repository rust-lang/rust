// This used to ICE because it creates an `impl Trait` that captures a
// hidden empty region.

//@ check-pass

fn server() -> impl FilterBase2 {
    segment2(|| { loop { } }).map2(|| "")
}

trait FilterBase2 {
    fn map2<F>(self, _fn: F) -> Map2<F> where Self: Sized { loop { } }
}

struct Map2<F> { _func: F }

impl<F> FilterBase2 for Map2<F> { }

fn segment2<F>(_fn: F) -> Map2<F> where F: Fn() -> Result<(), ()> {
    loop { }
}

fn main() { server(); }
