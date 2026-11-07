//! Functions with a mismatch between the expected and found type where the difference is a
//! reference may trigger analysis for additional help. In this test the expected type will be
//! &'a Container<&'a u8> and the found type will be Container<&'?0 u8>.
//!
//! This test exercises a scenario where the found type being analyzed contains an inference region
//! variable ('?0). This cannot be used in comparisons because the variable no longer exists by the
//! time the later analysis is performed.
//!
//! This is a regression test of #140823

trait MyFn<P> {}

struct Container<T> {
    data: T,
}

struct Desugared {
    callback: Box<dyn for<'a> MyFn<&'a Container<&'a u8>>>,
}

fn test(callback: Box<dyn for<'a> MyFn<Container<&'a u8>>>) -> Desugared {
    Desugared { callback }
    //~^ ERROR mismatched types
}

fn main() {}
