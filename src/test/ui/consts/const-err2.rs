// needed because negating int::MIN will behave differently between
// optimized compilation and unoptimized compilation and thus would
// lead to different lints being emitted

// revisions: noopt opt opt_with_overflow_checks
//[noopt]compile-flags: -C opt-level=0
//[opt]compile-flags: -O
//[opt_with_overflow_checks]compile-flags: -C overflow-checks=on -O

// build-fail

#![feature(rustc_attrs)]

fn black_box<T>(_: T) {
    unimplemented!()
}

fn main() {
    let a = -i8::MIN;
    //~^ ERROR arithmetic operation will overflow
    let a_i128 = -i128::MIN;
    //~^ ERROR arithmetic operation will overflow
    let b = 200u8 + 200u8 + 200u8;
    //~^ ERROR arithmetic operation will overflow
    let b_i128 = i128::MIN - i128::MAX;
    //~^ ERROR arithmetic operation will overflow
    let c = 200u8 * 4;
    //~^ ERROR arithmetic operation will overflow
    let d = 42u8 - (42u8 + 1);
    //~^ ERROR arithmetic operation will overflow
    let _e = [5u8][1];
    //~^ ERROR operation will panic
    black_box(a);
    black_box(a_i128);
    black_box(b);
    black_box(b_i128);
    black_box(c);
    black_box(d);
}
