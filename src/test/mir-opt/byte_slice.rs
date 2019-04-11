// compile-flags: -Z mir-opt-level=0

fn main() {
    let x = b"foo";
    let y = [5u8, b'x'];
}

// END RUST SOURCE
// START rustc.main.EraseRegions.after.mir
// ...
// _1 = const b"102111111";
// ...
// _2 = [const 5u8, const 120u8];
// ...
// END rustc.main.EraseRegions.after.mir
