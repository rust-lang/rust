//@ run-pass
//@ aux-build:static-methods-crate.rs

extern crate static_methods_crate;

use static_methods_crate::read;

pub fn main() {
    let result: isize = read("5".to_string());
    assert_eq!(result, 5);
    assert_eq!(read::readMaybe("false".to_string()), Some(false));
    assert_eq!(read::readMaybe("foo".to_string()), None::<bool>);
}
