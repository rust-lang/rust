//@ run-pass
#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn branch(vec: Vec<u32>) -> u32 {
    match vec {
        box [] => 0,
        box [1, _, 3] => 1,
        box [2, ..] => 2,
        _ => 1000,
    }
}

fn nested(vec: Vec<Vec<u32>>) -> u32 {
    match vec {
        box [box [], ..] => 1,
        box [box [0, ..], box [1, ..]] => 2,
        _ => 1000,
    }
}

fn main() {
    assert!(matches!(Vec::<u32>::new(), box []));
    assert!(matches!(vec![1], box [1]));
    assert!(matches!(&vec![1], box [1]));
    assert!(matches!(vec![&1], box [1]));
    assert!(matches!(vec![vec![1]], box [box [1]]));

    assert_eq!(branch(vec![]), 0);
    assert_eq!(branch(vec![1, 2, 3]), 1);
    assert_eq!(branch(vec![3, 2, 1]), 1000);
    assert_eq!(branch(vec![2]), 2);
    assert_eq!(branch(vec![2, 3]), 2);
    assert_eq!(branch(vec![3, 2]), 1000);

    assert_eq!(nested(vec![vec![], vec![2]]), 1);
    assert_eq!(nested(vec![vec![0], vec![1]]), 2);
    assert_eq!(nested(vec![vec![0, 2], vec![1, 2]]), 2);
}
