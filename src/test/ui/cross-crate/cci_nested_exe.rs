// run-pass
// aux-build:cci_nested_lib.rs


extern crate cci_nested_lib;
use cci_nested_lib::*;

pub fn main() {
    let lst = new_int_alist();
    alist_add(&lst, 22, "hi".to_string());
    alist_add(&lst, 44, "ho".to_string());
    assert_eq!(alist_get(&lst, 22), "hi".to_string());
    assert_eq!(alist_get(&lst, 44), "ho".to_string());

    let lst = new_int_alist_2();
    alist_add(&lst, 22, "hi".to_string());
    alist_add(&lst, 44, "ho".to_string());
    assert_eq!(alist_get(&lst, 22), "hi".to_string());
    assert_eq!(alist_get(&lst, 44), "ho".to_string());
}
