// Regression test for #123134. This is a variant of `trivial-bounds-sized.rs`
// that previously compiled without any feature gate and used to trigger a delayed bug.

trait Api: Sized {
    type Device: ?Sized;
}

struct OpenDevice<A: Api>
where
    A::Device: Sized,
{
    device: A::Device, // <- this is the type that ends up being unsized.
    queue: (),
}

trait Adapter {
    type A: Api;

    fn open() -> OpenDevice<Self::A>
    where
        <Self::A as Api>::Device: Sized;
}

struct ApiS;

impl Api for ApiS {
    type Device = [u8];
}

impl<T> Adapter for T {
    type A = ApiS;

    fn open() -> OpenDevice<Self::A>
    where
        <Self::A as Api>::Device: Sized,
        //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    {
        unreachable!()
    }
}

pub fn main() {}
