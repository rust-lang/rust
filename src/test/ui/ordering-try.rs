// run-pass

use std::cmp::Ordering;
use std::iter;

#[derive(Eq, PartialEq)]
struct Test(bool, u8, &'static str);

const BOOL_VALUES: [bool; 2] = [true, false];
const U8_VALUES: [u8; 3] = [1, 2, 3];
const STR_VALUES: [&str; 3] = ["a", "ab", "c"];

fn test_values() -> impl Iterator<Item = Test> {
    BOOL_VALUES.iter().flat_map(|&a| {
        U8_VALUES.iter().flat_map(move |&b| {
            STR_VALUES.iter().map(move |&c| Test(a, b, c))
        })
    })
}

// This Ord implementation should behave the same as #[derive(Ord)],
// but uses the Try operator on Ordering values
impl Ord for Test {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)?;
        self.1.cmp(&other.1)?;
        self.2.cmp(&other.2)
    }
}

impl PartialOrd for Test {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn test_struct_cmp() {
    for test1 in test_values() {
        let test1_alt = (test1.0, test1.1, test1.2);
        for test2 in test_values() {
            let test2_alt = (test2.0, test2.1, test2.2);
            assert_eq!(test1.cmp(&test2), test1_alt.cmp(&test2_alt));
        }
    }
}

// Implement Iterator::cmp() using the Try operator
fn cmp<A, I1, I2>(mut iter1: I1, mut iter2: I2) -> Ordering
where
    A: Ord,
    I1: Iterator<Item = A>,
    I2: Iterator<Item = A>,
{
    loop {
        match (iter1.next(), iter2.next()) {
            (Some(x), Some(y)) => x.cmp(&y)?,
            (x, y) => return x.cmp(&y),
        }
    }
}

fn u8_sequences() -> impl Iterator<Item = Vec<u8>> {
    iter::once(vec![])
        .chain(U8_VALUES.iter().map(|&a| vec![a]))
        .chain(U8_VALUES.iter().flat_map(|&a| {
            U8_VALUES.iter().map(move |&b| vec![a, b])
        }))
        .chain(U8_VALUES.iter().flat_map(|&a| {
            U8_VALUES.iter().flat_map(move |&b| {
                U8_VALUES.iter().map(move |&c| vec![a, b, c])
            })
        }))
}

fn test_slice_cmp() {
    for sequence1 in u8_sequences() {
        for sequence2 in u8_sequences() {
            assert_eq!(
                cmp(sequence1.iter().copied(), sequence2.iter().copied()),
                sequence1.iter().copied().cmp(sequence2.iter().copied()),
            );
        }
    }
}

fn main() {
    test_struct_cmp();
    test_slice_cmp();
}
