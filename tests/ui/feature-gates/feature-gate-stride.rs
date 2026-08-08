use std::mem;

fn main() {
    let tmp = mem::stride_of::<u8>();
    //~^ ERROR use of unstable library feature `stride`
}
