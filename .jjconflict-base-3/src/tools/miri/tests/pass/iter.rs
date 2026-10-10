fn iter_empty_and_zst() {
    // Iterate over a Unique::empty()
    for _ in Vec::<u32>::new().iter() {
        panic!("We should never be here.");
    }

    // Iterate over a ZST (uses arith_offset internally)
    let mut count = 0;
    for _ in &[(), (), ()] {
        count += 1;
    }
    assert_eq!(count, 3);
}

fn test_iterator_step_by_nth() {
    let mut it = (0..16).step_by(5);
    assert_eq!(it.nth(0), Some(0));
    assert_eq!(it.nth(0), Some(5));
    assert_eq!(it.nth(0), Some(10));
    assert_eq!(it.nth(0), Some(15));
    assert_eq!(it.nth(0), None);
}

fn iter_any() {
    let f = |x: &u8| 10u8 == *x;
    f(&1u8);

    let g = |(), x: &u8| 10u8 == *x;
    g((), &1u8);

    let h = |(), (), x: &u8| 10u8 == *x;
    h((), (), &1u8);

    [1, 2, 3u8].iter().any(|elt| 10 == *elt);
}

fn main() {
    test_iterator_step_by_nth();
    iter_any();
    iter_empty_and_zst();
}
