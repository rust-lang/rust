fn main() {
    for _ in Vec::<u32>::new().iter() { // this iterates over a Unique::empty()
        panic!("We should never be here.");
    }

    // Iterate over a ZST (uses arith_offset internally)
    let mut count = 0;
    for _ in &[(), (), ()] {
        count += 1;
    }
    assert_eq!(count, 3);
}
