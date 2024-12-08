//@ aux-build:hidden.rs

extern crate hidden;

use hidden::HiddenStruct;

struct InCrate {
    a: usize,
    b: bool,
    #[doc(hidden)]
    im_hidden: u8
}

fn main() {
    let HiddenStruct { one, two } = HiddenStruct::default();
    //~^ pattern requires `..` due to inaccessible fields

    let HiddenStruct { one } = HiddenStruct::default();
    //~^ pattern does not mention field `two` and inaccessible fields

    let HiddenStruct { one, hide } = HiddenStruct::default();
    //~^ pattern does not mention field `two`

    let InCrate { a, b } = InCrate { a: 0, b: false, im_hidden: 0 };
    //~^ pattern does not mention field `im_hidden`
}
