#![warn(clippy::needless_range_loop)]
#![allow(
    clippy::uninlined_format_args,
    clippy::unnecessary_literal_unwrap,
    clippy::useless_vec
)]

static STATIC: [usize; 4] = [0, 1, 8, 16];
const CONST: [usize; 4] = [0, 1, 8, 16];
const MAX_LEN: usize = 42;

fn main() {
    let mut vec = vec![1, 2, 3, 4];
    let vec2 = vec![1, 2, 3, 4];
    for i in 0..vec.len() {
        println!("{}", vec[i]);
    }

    for i in 0..vec.len() {
        let i = 42; // make a different `i`
        println!("{}", vec[i]); // ok, not the `i` of the for-loop
    }

    for i in 0..vec.len() {
        let _ = vec[i];
    }

    // ICE #746
    for j in 0..4 {
        println!("{:?}", STATIC[j]);
    }

    for j in 0..4 {
        println!("{:?}", CONST[j]);
    }

    for i in 0..vec.len() {
        println!("{} {}", vec[i], i);
    }
    for i in 0..vec.len() {
        // not an error, indexing more than one variable
        println!("{} {}", vec[i], vec2[i]);
    }

    for i in 0..vec.len() {
        println!("{}", vec2[i]);
    }

    for i in 5..vec.len() {
        println!("{}", vec[i]);
    }

    for i in 0..MAX_LEN {
        println!("{}", vec[i]);
    }

    for i in 0..=MAX_LEN {
        println!("{}", vec[i]);
    }

    for i in 5..10 {
        println!("{}", vec[i]);
    }

    for i in 5..=10 {
        println!("{}", vec[i]);
    }

    for i in 5..vec.len() {
        println!("{} {}", vec[i], i);
    }

    for i in 5..10 {
        println!("{} {}", vec[i], i);
    }

    // #2542
    for i in 0..vec.len() {
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
