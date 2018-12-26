#![allow(unused_assignments)]
#![allow(unknown_lints)]

#![allow(dead_assignment)]

pub fn main() {
    let x : String = "hello".to_string();
    let _y : String = "there".to_string();
    let mut z = "thing".to_string();
    z = x;
    assert_eq!(z.as_bytes()[0], ('h' as u8));
    assert_eq!(z.as_bytes()[4], ('o' as u8));
}
