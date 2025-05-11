#![allow(clippy::useless_vec)]
#[warn(clippy::range_zip_with_len)]
fn main() {
    let v1 = vec![1, 2, 3];
    let v2 = vec![4, 5];
    let _x = v1.iter().zip(0..v1.len());
    //~^ range_zip_with_len

    let _y = v1.iter().zip(0..v2.len()); // No error
}

#[allow(unused)]
fn no_panic_with_fake_range_types() {
    struct Range {
        foo: i32,
    }

    let _ = Range { foo: 0 };
}
