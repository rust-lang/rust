// compile-pass

#![feature(const_fn, const_fn_union)]

#![deny(const_err)]

union Bar {
    a: &'static u8,
    b: usize,
}

const fn bar() -> u8 {
    unsafe {
        // this will error as long as this test
        // is run on a system whose pointers need more
        // than 8 bits
        Bar { a: &42 }.b as u8
    }
}

fn main() {
    // FIXME(oli-obk): this should panic at runtime
    // this will actually compile, but then
    // abort at runtime (not panic, hard abort).
    let x: &'static u8 = &(bar() + 1);
    let y = *x;
    unreachable!();
}
