extern mod dylib;
extern mod both;

use std::cast;

fn main() {
    assert_eq!(unsafe { cast::transmute::<&int, uint>(&both::foo) },
               dylib::addr());
}
