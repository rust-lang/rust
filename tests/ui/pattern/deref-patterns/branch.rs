//@ revisions: explicit implicit
//@ run-pass
// Test the execution of deref patterns.
#![feature(deref_patterns)]
#![allow(incomplete_features)]

#[cfg(explicit)]
fn branch(vec: Vec<u32>) -> u32 {
    match vec {
        deref!([]) => 0,
        deref!([1, _, 3]) => 1,
        deref!([2, ..]) => 2,
        _ => 1000,
    }
}

#[cfg(implicit)]
fn branch(vec: Vec<u32>) -> u32 {
    match vec {
        [] => 0,
        [1, _, 3] => 1,
        [2, ..] => 2,
        _ => 1000,
    }
}

#[cfg(explicit)]
fn nested(vec: Vec<Vec<u32>>) -> u32 {
    match vec {
        deref!([deref!([]), ..]) => 1,
        deref!([deref!([0, ..]), deref!([1, ..])]) => 2,
        _ => 1000,
    }
}

#[cfg(implicit)]
fn nested(vec: Vec<Vec<u32>>) -> u32 {
    match vec {
        [[], ..] => 1,
        [[0, ..], [1, ..]] => 2,
        _ => 1000,
    }
}

fn main() {
    assert!(matches!(Vec::<u32>::new(), deref!([])));
    assert!(matches!(vec![1], deref!([1])));
    assert!(matches!(&vec![1], deref!([1])));
    assert!(matches!(vec![&1], deref!([1])));
    assert!(matches!(vec![vec![1]], deref!([deref!([1])])));

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
