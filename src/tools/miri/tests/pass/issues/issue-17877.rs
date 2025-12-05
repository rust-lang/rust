fn main() {
    assert_eq!(
        match [0u8; 16 * 1024] {
            _ => 42_usize,
        },
        42_usize,
    );

    assert_eq!(
        match [0u8; 16 * 1024] {
            [1, ..] => 0_usize,
            [0, ..] => 1_usize,
            _ => 2_usize,
        },
        1_usize,
    );
}
