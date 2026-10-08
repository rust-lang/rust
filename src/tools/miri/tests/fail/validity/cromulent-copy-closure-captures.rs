use core::num::NonZeroU8;

fn main() {
    let mut nonzero = NonZeroU8::new(1).unwrap();
    unsafe {
        (&raw mut nonzero).cast::<u8>().write(0);
    }
    let _ = || nonzero; //~ ERROR: constructing invalid value
}
