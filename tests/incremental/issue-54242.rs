//@ revisions: rpass cfail

trait Tr
where
    (Self::Arr,): Sized,
{
    type Arr;

    const C: usize = 0;
}

impl Tr for str {
    #[cfg(rpass)]
    type Arr = [u8; 8];
    #[cfg(cfail)]
    type Arr = [u8; Self::C];
    //[cfail]~^ ERROR cycle detected when caching mir
}

fn main() {}
