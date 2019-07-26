// run-pass
trait Lattice {
    const BOTTOM: Self;
}

impl<T> Lattice for Option<T> {
    const BOTTOM: Option<T> = None;
}

fn main(){}
