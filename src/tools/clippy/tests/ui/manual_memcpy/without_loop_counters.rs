#![warn(clippy::manual_memcpy)]
#![allow(clippy::assigning_clones, clippy::useless_vec, clippy::needless_range_loop)]

//@no-rustfix
const LOOP_OFFSET: usize = 5000;

pub fn manual_copy(src: &[i32], dst: &mut [i32], dst2: &mut [i32]) {
    // plain manual memcpy
    for i in 0..src.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        //~| NOTE: `-D clippy::manual-memcpy` implied by `-D warnings`
        dst[i] = src[i];
    }

    // dst offset memcpy
    for i in 0..src.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i + 10] = src[i];
    }

    // src offset memcpy
    for i in 0..src.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i + 10];
    }

    // src offset memcpy
    for i in 11..src.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i - 10];
    }

    // overwrite entire dst
    for i in 0..dst.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i];
    }

    // manual copy with branch - can't easily convert to memcpy!
    for i in 0..src.len() {
        dst[i] = src[i];
        if dst[i] > 5 {
            break;
        }
    }

    // multiple copies - suggest two memcpy statements
    for i in 10..256 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i - 5];
        dst2[i + 500] = src[i]
    }

    // this is a reversal - the copy lint shouldn't be triggered
    for i in 10..LOOP_OFFSET {
        dst[i + LOOP_OFFSET] = src[LOOP_OFFSET - i];
    }

    let some_var = 5;
    // Offset in variable
    for i in 10..LOOP_OFFSET {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i + LOOP_OFFSET] = src[i - some_var];
    }

    // Non continuous copy - don't trigger lint
    for i in 0..10 {
        dst[i + i] = src[i];
    }

    let src_vec = vec![1, 2, 3, 4, 5];
    let mut dst_vec = vec![0, 0, 0, 0, 0];

    // make sure vectors are supported
    for i in 0..src_vec.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst_vec[i] = src_vec[i];
    }

    // lint should not trigger when either
    // source or destination type is not
    // slice-like, like DummyStruct
    struct DummyStruct(i32);

    impl ::std::ops::Index<usize> for DummyStruct {
        type Output = i32;

        fn index(&self, _: usize) -> &i32 {
            &self.0
        }
    }

    let src = DummyStruct(5);
    let mut dst_vec = vec![0; 10];

    for i in 0..10 {
        dst_vec[i] = src[i];
    }

    // Simplify suggestion (issue #3004)
    let src = [0, 1, 2, 3, 4];
    let mut dst = [0, 0, 0, 0, 0, 0];
    let from = 1;

    for i in from..from + src.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i - from];
    }

    for i in from..from + 3 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i - from];
    }

    #[allow(clippy::identity_op)]
    for i in 0..5 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i - 0] = src[i];
    }

    #[allow(clippy::reversed_empty_ranges)]
    for i in 0..0 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i];
    }

    // `RangeTo` `for` loop - don't trigger lint
    for i in 0.. {
        dst[i] = src[i];
    }

    // VecDeque - ideally this would work, but would require something like `range_as_slices`
    let mut dst = std::collections::VecDeque::from_iter([0; 5]);
    let src = std::collections::VecDeque::from_iter([0, 1, 2, 3, 4]);
    for i in 0..dst.len() {
        dst[i] = src[i];
    }
    let src = vec![0, 1, 2, 3, 4];
    for i in 0..dst.len() {
        dst[i] = src[i];
    }

    // Range is equal to array length
    let src = [0, 1, 2, 3, 4];
    let mut dst = [0; 4];
    for i in 0..4 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i];
    }

    let mut dst = [0; 6];
    for i in 0..5 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i];
    }

    let mut dst = [0; 5];
    for i in 0..5 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i];
    }

    // Don't trigger lint for following multi-dimensional arrays
    let src = [[0; 5]; 5];
    for i in 0..4 {
        dst[i] = src[i + 1][i];
    }
    for i in 0..5 {
        dst[i] = src[i][i];
    }
    for i in 0..5 {
        dst[i] = src[i][3];
    }

    let src = [0; 5];
    let mut dst = [[0; 5]; 5];
    for i in 0..5 {
        dst[i][i] = src[i];
    }

    let src = [[[0; 5]; 5]; 5];
    let mut dst = [0; 5];
    for i in 0..5 {
        dst[i] = src[i][i][i];
    }
    for i in 0..5 {
        dst[i] = src[i][i][0];
    }
    for i in 0..5 {
        dst[i] = src[i][0][i];
    }
    for i in 0..5 {
        dst[i] = src[0][i][i];
    }
    for i in 0..5 {
        dst[i] = src[0][i][1];
    }
    for i in 0..5 {
        dst[i] = src[i][0][1];
    }

    // Trigger lint
    let src = [[0; 5]; 5];
    let mut dst = [0; 5];
    for i in 0..5 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[0][i];
    }

    let src = [[[0; 5]; 5]; 5];
    for i in 0..5 {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[0][1][i];
    }
}

#[warn(clippy::needless_range_loop, clippy::manual_memcpy)]
pub fn manual_clone(src: &[String], dst: &mut [String]) {
    for i in 0..src.len() {
        //~^ ERROR: it looks like you're manually copying between slices
        dst[i] = src[i].clone();
    }
}

fn main() {}
