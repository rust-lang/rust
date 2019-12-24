#![feature(const_transmute)]

fn main() {
    match &b""[..] {
        ZST => {}
        //~^ ERROR could not evaluate constant pattern
    }
}

const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
//~^ ERROR any use of this value will cause an error
//~| ERROR cannot transmute between types of different sizes

// Once the `any use of this value will cause an error` disappears in this test, make sure to
// remove the `TransmuteSizeDiff` error variant and make its emitter site an assertion again.
