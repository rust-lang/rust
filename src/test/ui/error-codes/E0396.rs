// gate-test-const_raw_ptr_deref

const REG_ADDR: *const u8 = 0x5f3759df as *const u8;

const VALUE: u8 = unsafe { *REG_ADDR };
//~^ ERROR dereferencing raw pointers in constants is unstable

const unsafe fn unreachable() -> ! {
    use std::convert::Infallible;

    const INFALLIBLE: *const Infallible = [].as_ptr();
    match *INFALLIBLE {}
    //~^ ERROR dereferencing raw pointers in constant functions is unstable

    const BAD: () = unsafe { match *INFALLIBLE {} };
    //~^ ERROR dereferencing raw pointers in constants is unstable
}

fn main() {
}
