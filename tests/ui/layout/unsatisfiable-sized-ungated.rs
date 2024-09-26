//@ check-pass
// issue: #123134

//! This is a variant of `trivial-bounds-sized.rs` that compiles without any
//! feature gates and used to trigger a delayed bug.

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
        // ^ the bound expands to `<<T as Adapter>::A as Api>::Device: Sized`, which
        // is not considered trivial due to containing the type parameter `T`
    {
        unreachable!()
    }
}

pub fn main() {}
