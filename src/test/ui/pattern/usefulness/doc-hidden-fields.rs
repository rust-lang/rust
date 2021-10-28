// aux-build:hidden.rs

extern crate hidden;

use hidden::HiddenStruct;

fn main() {
    let HiddenStruct { one, two, } = HiddenStruct::default();
    //~^ pattern requires `..` due to inaccessible fields

    let HiddenStruct { one, } = HiddenStruct::default();
    //~^ pattern does not mention field `two` and inaccessible fields

    let HiddenStruct { one, hide } = HiddenStruct::default();
    //~^ pattern does not mention field `two`
}
