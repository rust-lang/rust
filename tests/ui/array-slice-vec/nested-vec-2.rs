// run-pass

// Test that using the `vec!` macro nested within itself works
// when the contents implement Drop

struct D(u32);

impl Drop for D {
    fn drop(&mut self) { println!("Dropping {}", self.0); }
}

fn main() {
    let nested = vec![vec![D(1u32), D(2u32), D(3u32)]];
    assert_eq!(nested[0][1].0, 2);
}
