// run-pass

// Test that using the `vec!` macro nested within itself works

fn main() {
    let nested = vec![vec![1u32, 2u32, 3u32]];
    assert_eq!(nested[0][1], 2);
}
