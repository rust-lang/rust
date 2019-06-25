#![feature(rustc_attrs)]
#![deny(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

fn main() {
    let b = 200u8 + 200u8 + 200u8;
    //~^ ERROR const_err
    let c = 200u8 * 4;
    //~^ ERROR const_err
    let d = 42u8 - (42u8 + 1);
    //~^ ERROR const_err
    let _e = [5u8][1];
    //~^ ERROR const_err
    black_box(b);
    black_box(c);
    black_box(d);
}
