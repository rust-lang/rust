// revisions: rpass cfail

trait Tr {
    type Arr;

    const C: usize = 0;
}

impl Tr for str {
    #[cfg(rpass)]
    type Arr = [u8; 8];
    #[cfg(cfail)]
    type Arr = [u8; Self::C];
    //[cfail]~^ ERROR cycle detected when const-evaluating
}

fn main() {}
