//@no-rustfix
#![allow(clippy::useless_vec)]
#[warn(clippy::range_zip_with_len)]
fn main() {
    let v1: Vec<u64> = vec![1, 2, 3];
    let v2: Vec<u64> = vec![4, 5];

    // Do not autofix, `filter()` would not consume the iterator.
    //~v range_zip_with_len
    v1.iter().zip(0..v1.len()).filter(|(_, i)| *i < 2).for_each(|(e, i)| {
        let _: &u64 = e;
        let _: usize = i;
    });
}
