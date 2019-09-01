// needed because negating int::MIN will behave differently between
// optimized compilation and unoptimized compilation and thus would
// lead to different lints being emitted
// compile-flags: -O

#![feature(rustc_attrs)]
#![allow(exceeding_bitshifts)]
#![deny(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

fn main() {
    let a = -std::i8::MIN;
    //~^ ERROR const_err
    let b = 200u8 + 200u8 + 200u8;
    //~^ ERROR const_err
    let c = 200u8 * 4;
    //~^ ERROR const_err
    let d = 42u8 - (42u8 + 1);
    //~^ ERROR const_err
    let _e = [5u8][1];
    //~^ ERROR const_err
    black_box(a);
    black_box(b);
    black_box(c);
    black_box(d);
}
