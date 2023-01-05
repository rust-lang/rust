const REG_ADDR: *mut u8 = 0x5f3759df as *mut u8;

const VALUE: u8 = unsafe { *REG_ADDR };
//~^ ERROR dereferencing raw mutable pointers in constants is unstable

const unsafe fn unreachable() -> ! {
    use std::convert::Infallible;

    const INFALLIBLE: *mut Infallible = &[] as *const [Infallible] as *const _ as _;
    match *INFALLIBLE {}
    //~^ ERROR dereferencing raw mutable pointers in constant functions is unstable

    const BAD: () = unsafe { match *INFALLIBLE {} };
    //~^ ERROR dereferencing raw mutable pointers in constants is unstable
}

fn main() {
}
