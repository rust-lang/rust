// Check that specialization must be ungated to use the `default` keyword

// gate-test-specialization

trait Foo {
    fn foo(&self);
}

#[cfg(FALSE)]
impl<T> Foo for T {
    default //~ ERROR specialization is unstable
    fn foo(&self) {}
}

#[cfg(FALSE)]
default //~ ERROR specialization is unstable
impl<T> Foo for T {
    fn foo(&self) {}
}

fn main() {}
