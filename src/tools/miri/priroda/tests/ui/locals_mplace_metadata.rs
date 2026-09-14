#![feature(unsized_fn_params)]
#![allow(incomplete_features, internal_features, unused_variables)]

trait Marker {}

impl Marker for (i32, i32) {}

fn slice_param(slice: [u8]) {
    std::hint::black_box(&slice);
}

fn str_param(text: str) {
    std::hint::black_box(&text);
}

fn dyn_param(value: dyn Marker) {
    std::hint::black_box(&value);
}

fn main() {
    let slice: Box<[u8]> = Box::new([1_u8, 2, 3]);
    slice_param(*slice);
    str_param(*String::from("abc").into_boxed_str());
    let value: Box<dyn Marker> = Box::new((1_i32, 2_i32));
    dyn_param(*value);
}
