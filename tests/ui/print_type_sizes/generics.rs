//@ compile-flags: -Z print-type-sizes --crate-type=lib
//@ build-pass
//@ ignore-pass
// ^-- needed because `--pass check` does not emit the output needed.
//     FIXME: consider using an attribute instead of side-effects.

// This file illustrates how generics are handled: types have to be
// monomorphized, in the MIR of the original function in which they
// occur, to have their size reported.

#[derive(Copy, Clone)]
pub struct Pair<T> {
    _car: T,
    _cdr: T,
}

impl<T> Pair<T> {
    fn new(a: T, d: T) -> Self {
        Pair {
            _car: a,
            _cdr: d,
        }
    }
}

#[derive(Copy, Clone)]
pub struct SevenBytes([u8; 7]);
pub struct FiftyBytes([u8; 50]);

pub struct ZeroSized;

impl SevenBytes {
    fn new() -> Self { SevenBytes([0; 7]) }
}

impl FiftyBytes {
    fn new() -> Self { FiftyBytes([0; 50]) }
}

pub fn f1<T:Copy>(x: T) {
    let _v: Pair<T> = Pair::new(x, x);
    let _v2: Pair<FiftyBytes> =
        Pair::new(FiftyBytes::new(), FiftyBytes::new());
}

pub fn start() {
    let _b: Pair<u8> = Pair::new(0, 0);
    let _s: Pair<SevenBytes> = Pair::new(SevenBytes::new(), SevenBytes::new());
    let ref _z: ZeroSized = ZeroSized;
    f1::<SevenBytes>(SevenBytes::new());
}
