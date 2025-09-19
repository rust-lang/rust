#![allow(clippy::useless_vec)]
#[warn(clippy::range_zip_with_len)]
fn main() {
    let v1: Vec<u64> = vec![1, 2, 3];
    let v2: Vec<u64> = vec![4, 5];
    let _x = v1.iter().zip(0..v1.len());
    //~^ range_zip_with_len

    //~v range_zip_with_len
    for (e, i) in v1.iter().zip(0..v1.len()) {
        let _: &u64 = e;
        let _: usize = i;
    }

    //~v range_zip_with_len
    v1.iter().zip(0..v1.len()).for_each(|(e, i)| {
        let _: &u64 = e;
        let _: usize = i;
    });

    let _y = v1.iter().zip(0..v2.len()); // No error
}

#[allow(unused)]
fn no_panic_with_fake_range_types() {
    struct Range {
        foo: i32,
    }

    let _ = Range { foo: 0 };
}
