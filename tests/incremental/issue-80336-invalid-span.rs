// Regression test for issue #80336
// Test that we properly handle encoding, decoding, and hashing
// of spans with an invalid location and non-root `SyntaxContext`

//@ revisions:rpass1 rpass2
//@ only-x86_64

pub fn main() {
    let _ = is_x86_feature_detected!("avx2");
}
