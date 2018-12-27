#![allow(non_camel_case_types)]
// compile-flags: --edition 2015

fn main() {
    let try = 2;
    struct try { try: u32 };
    let try: try = try { try };
    assert_eq!(try.try, 2);
}
