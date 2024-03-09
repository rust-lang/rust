//@ run-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn simple_vec(vec: Vec<u32>) -> u32 {
    match vec {
        box [] => 100,
        // FIXME(deref_patterns): fake borrows break guards
        // box [x] if x == 4 => x + 4,
        box [x] => x,
        box [1, x] => x + 200,
        box ref slice => slice.iter().sum(),
        _ => 2000,
    }
}

fn nested_vec(vecvec: Vec<Vec<u32>>) -> u32 {
    match vecvec {
        box [] => 0,
        box [box [x]] => x,
        box [box [0, x] | box [1, x]] => x,
        box [ref x] => x.iter().sum(),
        box [box [], box [1, x, y]] => y - x,
        _ => 2000,
    }
}

fn main() {
    assert_eq!(simple_vec(vec![1]), 1);
    assert_eq!(simple_vec(vec![1, 2]), 202);
    assert_eq!(simple_vec(vec![1, 2, 3]), 6);

    assert_eq!(nested_vec(vec![vec![0, 42]]), 42);
    assert_eq!(nested_vec(vec![vec![1, 42]]), 42);
    assert_eq!(nested_vec(vec![vec![1, 2, 3]]), 6);
    assert_eq!(nested_vec(vec![vec![], vec![1, 2, 3]]), 1);
}
