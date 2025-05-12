fn main() {
    use std::mem::{transmute, swap};
    let a = 1;
    let b = 2;
    unsafe {swap::<&mut _>(transmute(&a), transmute(&b))};
    //~^ ERROR type annotations needed
}
