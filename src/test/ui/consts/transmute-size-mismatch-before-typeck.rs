#![feature(const_transmute)]

// normalize-stderr-64bit "64 bits" -> "word size"
// normalize-stderr-32bit "32 bits" -> "word size"
// normalize-stderr-64bit "128 bits" -> "2 * word size"
// normalize-stderr-32bit "64 bits" -> "2 * word size"

fn main() {
    match &b""[..] {
        ZST => {} //~ ERROR could not evaluate constant pattern
                  //~| ERROR could not evaluate constant pattern
    }
}

const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
//~^ ERROR any use of this value will cause an error
//~| ERROR cannot transmute between types of different sizes

// Once the `any use of this value will cause an error` disappears in this test, make sure to
// remove the `TransmuteSizeDiff` error variant and make its emitter site an assertion again.
