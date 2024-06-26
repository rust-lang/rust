//@ run-pass
//@ known-bug: #107975

fn main() {
    let a: usize = {
        let v = 0u8;
        &v as *const _ as usize
    };
    let b: usize = {
        let v = 0u8;
        &v as *const _ as usize
    };
    let i: usize = a - b;
    assert_ne!(i, 0);
    println!("{}", i);
    assert_eq!(i, 0);
}
