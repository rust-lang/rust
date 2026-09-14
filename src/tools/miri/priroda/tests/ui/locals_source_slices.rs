#![feature(unsized_fn_params)]
#![allow(incomplete_features, internal_features, unused_variables)]

fn slice_zero(slice: [u8]) {
    std::hint::black_box(&slice);
}

fn slice_one(slice: [u8]) {
    std::hint::black_box(&slice);
}

fn slice_many(slice: [u8]) {
    std::hint::black_box(&slice);
}

fn main() {
    slice_zero(*Box::<[u8]>::from([]));
    slice_one(*Box::<[u8]>::from([0x1a_u8]));
    slice_many(*Box::<[u8]>::from([0x1b_u8, 0x1c]));
}
