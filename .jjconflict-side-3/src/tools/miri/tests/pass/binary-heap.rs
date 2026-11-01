use std::collections::BinaryHeap;

fn zero_sized_push() {
    const N: usize = 8;

    for len in 0..N {
        let mut tester = BinaryHeap::with_capacity(len);
        assert_eq!(tester.len(), 0);
        assert!(tester.capacity() >= len);
        for _ in 0..len {
            tester.push(());
        }
        assert_eq!(tester.len(), len);
        assert_eq!(tester.iter().count(), len);
        tester.clear();
    }
}

fn drain() {
    let mut heap = (0..128i32).collect::<BinaryHeap<_>>();

    assert!(!heap.is_empty());

    let mut sum = 0;
    for x in heap.drain() {
        sum += x;
    }
    assert_eq!(sum, 127 * 128 / 2);

    assert!(heap.is_empty());
}

fn main() {
    zero_sized_push();
    drain();
}
