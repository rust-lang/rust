#![allow(clippy::cognitive_complexity)]

static STATIC: [usize; 4] = [0, 1, 8, 16];
const CONST: [usize; 4] = [0, 1, 8, 16];

fn calc_idx(i: usize) -> usize {
    (i + i + 20) % 4
}

fn main() {
    const MAX_LEN: usize = 42;

    let ns = vec![2, 3, 5, 7];

    for i in 3..10 {
        println!("{}", ns[i]);
    }

    for i in 3..10 {
        println!("{}", ns[i % 4]);
    }

    for i in 3..10 {
        println!("{}", ns[i % ns.len()]);
    }

    for i in 3..10 {
        println!("{}", ns[calc_idx(i)]);
    }

    for i in 3..10 {
        println!("{}", ns[calc_idx(i) % 4]);
    }

    let mut ms = vec![1, 2, 3, 4, 5, 6];
    for i in 0..ms.len() {
        ms[i] *= 2;
    }
    assert_eq!(ms, vec![2, 4, 6, 8, 10, 12]);

    let mut ms = vec![1, 2, 3, 4, 5, 6];
    for i in 0..ms.len() {
        let x = &mut ms[i];
        *x *= 2;
    }
    assert_eq!(ms, vec![2, 4, 6, 8, 10, 12]);

    let g = vec![1, 2, 3, 4, 5, 6];
    let glen = g.len();
    for i in 0..glen {
        let x: u32 = g[i + 1..].iter().sum();
        println!("{}", g[i] + x);
    }
    assert_eq!(g, vec![20, 18, 15, 11, 6, 0]);

    let mut g = vec![1, 2, 3, 4, 5, 6];
    let glen = g.len();
    for i in 0..glen {
        g[i] = g[i + 1..].iter().sum();
    }
    assert_eq!(g, vec![20, 18, 15, 11, 6, 0]);

    let x = 5;
    let mut vec = vec![0; 9];

    for i in x..x + 4 {
        vec[i] += 1;
    }

    let x = 5;
    let mut vec = vec![0; 10];

    for i in x..=x + 4 {
        vec[i] += 1;
    }

    let arr = [1, 2, 3];

    for i in 0..3 {
        println!("{}", arr[i]);
    }

    for i in 0..2 {
        println!("{}", arr[i]);
    }

    for i in 1..3 {
        println!("{}", arr[i]);
    }

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
