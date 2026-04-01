//@ check-pass
//@ compile-flags: -Z annotate-moves=100

// Test that valid annotate-moves flags are accepted

#[derive(Clone)]
struct TestStruct {
    data: [u64; 20], // 160 bytes
}

fn main() {
    let s = TestStruct { data: [42; 20] };
    let _copy = s.clone();
    let _moved = s;
}
