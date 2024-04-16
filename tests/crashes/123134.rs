//@ known-bug: #123134
trait Api: Sized {
    type Device: ?Sized;
}

struct OpenDevice<A: Api>
where
    A::Device: Sized,
{
    device: A::Device,
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
    {
        unreachable!()
    }
}

pub fn main() {}
