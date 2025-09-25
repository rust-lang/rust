#![warn(clippy::needless_range_loop)]
#![allow(
    clippy::uninlined_format_args,
    clippy::unnecessary_literal_unwrap,
    clippy::useless_vec,
    clippy::manual_slice_fill
)]
//@no-rustfix
static STATIC: [usize; 4] = [0, 1, 8, 16];
const CONST: [usize; 4] = [0, 1, 8, 16];
const MAX_LEN: usize = 42;

fn main() {
    let mut vec = vec![1, 2, 3, 4];
    let vec2 = vec![1, 2, 3, 4];
    for i in 0..vec.len() {
        //~^ needless_range_loop

        println!("{}", vec[i]);
    }

    for i in 0..vec.len() {
        let i = 42; // make a different `i`
        println!("{}", vec[i]); // ok, not the `i` of the for-loop
    }

    for i in 0..vec.len() {
        //~^ needless_range_loop

        let _ = vec[i];
    }

    // ICE #746
    for j in 0..4 {
        //~^ needless_range_loop

        println!("{:?}", STATIC[j]);
    }

    for j in 0..4 {
        //~^ needless_range_loop

        println!("{:?}", CONST[j]);
    }

    for i in 0..vec.len() {
        //~^ needless_range_loop

        println!("{} {}", vec[i], i);
    }
    for i in 0..vec.len() {
        // not an error, indexing more than one variable
        println!("{} {}", vec[i], vec2[i]);
    }

    for i in 0..vec.len() {
        //~^ needless_range_loop

        println!("{}", vec2[i]);
    }

    for i in 5..vec.len() {
        //~^ needless_range_loop

        println!("{}", vec[i]);
    }

    for i in 0..MAX_LEN {
        //~^ needless_range_loop

        println!("{}", vec[i]);
    }

    for i in 0..=MAX_LEN {
        //~^ needless_range_loop

        println!("{}", vec[i]);
    }

    for i in 5..10 {
        //~^ needless_range_loop

        println!("{}", vec[i]);
    }

    for i in 5..=10 {
        //~^ needless_range_loop

        println!("{}", vec[i]);
    }

    for i in 5..vec.len() {
        //~^ needless_range_loop

        println!("{} {}", vec[i], i);
    }

    for i in 5..10 {
        //~^ needless_range_loop

        println!("{} {}", vec[i], i);
    }

    // #2542
    for i in 0..vec.len() {
        //~^ needless_range_loop

        vec[i] = Some(1).unwrap_or_else(|| panic!("error on {}", i));
    }

    // #3788
    let test = Test {
        inner: vec![1, 2, 3, 4],
    };
    for i in 0..2 {
        println!("{}", test[i]);
    }

    // See #601
    for i in 0..10 {
        // no error, id_col does not exist outside the loop
        let mut id_col = [0f64; 10];
        id_col[i] = 1f64;
    }

    fn f<T>(_: &T, _: &T) -> bool {
        unimplemented!()
    }
    fn g<T>(_: &mut [T], _: usize, _: usize) {
        unimplemented!()
    }
    for i in 1..vec.len() {
        if f(&vec[i - 1], &vec[i]) {
            g(&mut vec, i - 1, i);
        }
    }

    for mid in 1..vec.len() {
        let (_, _) = vec.split_at(mid);
    }
}

struct Test {
    inner: Vec<usize>,
}

impl std::ops::Index<usize> for Test {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

fn partition<T: PartialOrd + Send>(v: &mut [T]) -> usize {
    let pivot = v.len() - 1;
    let mut i = 0;
    for j in 0..pivot {
        if v[j] <= v[pivot] {
            v.swap(i, j);
            i += 1;
        }
    }
    v.swap(i, pivot);
    i
}

pub fn manual_copy_same_destination(dst: &mut [i32], d: usize, s: usize) {
    // Same source and destination - don't trigger lint
    for i in 0..dst.len() {
        dst[d + i] = dst[s + i];
    }
}

mod issue_2496 {
    pub trait Handle {
        fn new_for_index(index: usize) -> Self;
        fn index(&self) -> usize;
    }

    pub fn test<H: Handle>() -> H {
        for x in 0..5 {
            let next_handle = H::new_for_index(x);
            println!("{}", next_handle.index());
        }
        unimplemented!()
    }
}

fn needless_loop() {
    use std::hint::black_box;
    let x = [0; 64];
    for i in 0..64 {
        let y = [0; 64];

        black_box(x[i]);
        black_box(y[i]);
    }

    for i in 0..64 {
        black_box(x[i]);
        black_box([0; 64][i]);
    }

    for i in 0..64 {
        black_box(x[i]);
        black_box([1, 2, 3, 4, 5, 6, 7, 8][i]);
    }

    for i in 0..64 {
        black_box([1, 2, 3, 4, 5, 6, 7, 8][i]);
    }
}

fn issue_15068() {
    let a = vec![vec![0u8; MAX_LEN]; MAX_LEN];
    let b = vec![0u8; MAX_LEN];

    for i in 0..MAX_LEN {
        // no error
        let _ = a[0][i];
        let _ = b[i];
    }

    for i in 0..MAX_LEN {
        // no error
        let _ = a[i][0];
        let _ = b[i];
    }

    for i in 0..MAX_LEN {
        // no error
        let _ = a[i][b[i] as usize];
    }

    for i in 0..MAX_LEN {
        //~^ needless_range_loop
        let _ = a[i][i];
    }

    for i in 0..MAX_LEN {
        //~^ needless_range_loop
        let _ = a[0][i];
    }
}
