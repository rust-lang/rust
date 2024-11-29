use std::fmt::Debug;

use super::sort_and_cancel;

fn flatten<T>(input: Vec<Option<T>>) -> Vec<T> {
    input.into_iter().flatten().collect()
}

fn sort_and_cancel_and_flatten<T: Clone + Ord>(pos: Vec<T>, neg: Vec<T>) -> (Vec<T>, Vec<T>) {
    let (pos_actual, neg_actual) = sort_and_cancel(pos, neg);
    (flatten(pos_actual), flatten(neg_actual))
}

#[track_caller]
fn check_test_case<T: Clone + Debug + Ord>(
    pos: Vec<T>,
    neg: Vec<T>,
    pos_expected: Vec<T>,
    neg_expected: Vec<T>,
) {
    eprintln!("pos = {pos:?}; neg = {neg:?}");
    let output = sort_and_cancel_and_flatten(pos, neg);
    assert_eq!(output, (pos_expected, neg_expected));
}

#[test]
fn cancellation() {
    let cases: &[(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)] = &[
        (vec![], vec![], vec![], vec![]),
        (vec![4, 2, 1, 5, 3], vec![], vec![1, 2, 3, 4, 5], vec![]),
        (vec![5, 5, 5, 5, 5], vec![5], vec![5, 5, 5, 5], vec![]),
        (vec![1, 1, 2, 2, 3, 3], vec![1, 2, 3], vec![1, 2, 3], vec![]),
        (vec![1, 1, 2, 2, 3, 3], vec![2, 4, 2], vec![1, 1, 3, 3], vec![4]),
    ];

    for (pos, neg, pos_expected, neg_expected) in cases {
        check_test_case(pos.to_vec(), neg.to_vec(), pos_expected.to_vec(), neg_expected.to_vec());
        // Same test case, but with its inputs flipped and its outputs flipped.
        check_test_case(neg.to_vec(), pos.to_vec(), neg_expected.to_vec(), pos_expected.to_vec());
    }
}
