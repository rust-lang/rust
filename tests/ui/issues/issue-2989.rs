//@ run-pass
#![allow(non_camel_case_types)]

trait methods { //~ WARN trait `methods` is never used
    fn to_bytes(&self) -> Vec<u8>;
}

impl methods for () {
    fn to_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
}

// the position of this function is significant! - if it comes before methods
// then it works, if it comes after it then it doesn't!
fn to_bools(bitv: Storage) -> Vec<bool> {
    (0..8).map(|i| {
        let w = i / 64;
        let b = i % 64;
        let x = 1 & (bitv.storage[w] >> b);
        x == 1
    }).collect()
}

struct Storage { storage: Vec<u64> }

pub fn main() {
    let bools = vec![false, false, true, false, false, true, true, false];
    let bools2 = to_bools(Storage{storage: vec![0b01100100]});

    for i in 0..8 {
        println!("{} => {} vs {}", i, bools[i], bools2[i]);
    }

    assert_eq!(bools, bools2);
}
