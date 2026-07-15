// Regression test for issue #84195
// Checks that we properly fire lints that occur inside
// anon consts.

#![deny(unused_attributes)]

macro_rules! len2 {
    () => { };
}

macro_rules! len {
    () => { { #[inline] len2!(); 0 } }; //~ ERROR `#[inline]` attribute cannot be used on macro calls
                                        //~| WARN this was previously accepted
}

fn main() {
    let val: [u8; len!()] = [];
}
