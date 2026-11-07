//@ revisions: rpass bfail

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
    #[cfg(bfail)]
    type Arr = [u8; Self::C];
    //[bfail]~^ ERROR cycle detected when
}

fn main() {}
