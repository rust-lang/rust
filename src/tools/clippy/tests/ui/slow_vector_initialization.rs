use std::iter::repeat;

fn main() {
    resize_vector();
    extend_vector();
    mixed_extend_resize_vector();
}

fn extend_vector() {
    // Extend with constant expression
    let len = 300;
    let mut vec1 = Vec::with_capacity(len);
    vec1.extend(repeat(0).take(len));

    // Extend with len expression
    let mut vec2 = Vec::with_capacity(len - 10);
    vec2.extend(repeat(0).take(len - 10));

    // Extend with mismatching expression should not be warned
    let mut vec3 = Vec::with_capacity(24322);
    vec3.extend(repeat(0).take(2));
}

fn mixed_extend_resize_vector() {
    // Mismatching len
    let mut mismatching_len = Vec::with_capacity(30);
    mismatching_len.extend(repeat(0).take(40));

    // Slow initialization
    let mut resized_vec = Vec::with_capacity(30);
    resized_vec.resize(30, 0);

    let mut extend_vec = Vec::with_capacity(30);
    extend_vec.extend(repeat(0).take(30));
}

fn resize_vector() {
    // Resize with constant expression
    let len = 300;
    let mut vec1 = Vec::with_capacity(len);
    vec1.resize(len, 0);

    // Resize mismatch len
    let mut vec2 = Vec::with_capacity(200);
    vec2.resize(10, 0);

    // Resize with len expression
    let mut vec3 = Vec::with_capacity(len - 10);
    vec3.resize(len - 10, 0);

    // Reinitialization should be warned
    vec1 = Vec::with_capacity(10);
    vec1.resize(10, 0);
}

fn do_stuff(vec: &mut [u8]) {}

fn extend_vector_with_manipulations_between() {
    let len = 300;
    let mut vec1: Vec<u8> = Vec::with_capacity(len);
    do_stuff(&mut vec1);
    vec1.extend(repeat(0).take(len));
}
