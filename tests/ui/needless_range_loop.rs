#![warn(clippy::needless_range_loop)]

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
