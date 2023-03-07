// Regression test for #88609:
// The return type for `main` is not normalized while checking if it implements
// the trait `std::process::Termination`.

// build-pass

trait Same {
    type Output;
}

impl<T> Same for T {
    type Output = T;
}

type Unit = <() as Same>::Output;

fn main() -> Result<Unit, std::io::Error> {
    unimplemented!()
}
