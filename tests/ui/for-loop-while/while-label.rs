// run-pass
#![allow(unreachable_code)]


pub fn main() {
    let mut i = 100;
    'w: while 1 + 1 == 2 {
        i -= 1;
        if i == 95 {
            break 'w;
            panic!("Should have broken out of loop");
        }
    }
    assert_eq!(i, 95);
}
