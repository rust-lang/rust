//@ run-pass

use std::collections::BinaryHeap;

fn make_pq() -> BinaryHeap<isize> {
    BinaryHeap::from(vec![1,2,3])
}

pub fn main() {
    let mut pq = make_pq();
    let mut sum = 0;
    while let Some(x) = pq.pop() {
        sum += x;
    }
    assert_eq!(sum, 6);

    pq = make_pq();
    sum = 0;
    'a: while let Some(x) = pq.pop() {
        sum += x;
        if x == 2 {
            break 'a;
        }
    }
    assert_eq!(sum, 5);

    pq = make_pq();
    sum = 0;
    'a2: while let Some(x) = pq.pop() {
        if x == 3 {
            continue 'a2;
        }
        sum += x;
    }
    assert_eq!(sum, 3);

    let mut pq1 = make_pq();
    sum = 0;
    while let Some(x) = pq1.pop() {
        let mut pq2 = make_pq();
        while let Some(y) = pq2.pop() {
            sum += x * y;
        }
    }
    assert_eq!(sum, 6 + 12 + 18);
}
