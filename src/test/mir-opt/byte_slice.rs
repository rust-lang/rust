// compile-flags: -Z mir-opt-level=0

fn main() {
    let x = b"foo";
    let y = [5u8, b'x'];
}

// END RUST SOURCE
// START rustc.main.EraseRegions.after.mir
// ...
// _1 = const Scalar(Ptr(Pointer { alloc_id: AllocId(0), offset: Size { raw: 0 }, tag: () })) : &[u8; 3];
// ...
// _2 = [const 5u8, const 120u8];
// ...
// END rustc.main.EraseRegions.after.mir
