#![deny(not_cenum_cast)]

enum E {
    A(),
    B{},
    C,
}

fn main() {
    let i = E::A() as u32;
    //~^ ERROR cannot cast enum `E` into integer `u32` because it is not C-like
    //~| WARN this was previously accepted
}
