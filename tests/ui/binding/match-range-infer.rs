// run-pass
// Test that type inference for range patterns works correctly (is bi-directional).

pub fn main() {
    match 1 {
        1 ..= 3 => {}
        _ => panic!("should match range")
    }
    match 1 {
        1 ..= 3u16 => {}
        _ => panic!("should match range with inferred start type")
    }
    match 1 {
        1u16 ..= 3 => {}
        _ => panic!("should match range with inferred end type")
    }
}
