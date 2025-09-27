//@ check-pass
//@ compile-flags: -Z instrument-moves -Z instrument-moves-size-limit=100

// Test that valid instrument-moves flags are accepted

#[derive(Clone)]
struct TestStruct {
    data: [u64; 20], // 160 bytes
}

fn main() {
    let s = TestStruct { data: [42; 20] };
    let _copy = s.clone();
    let _moved = s;
}
