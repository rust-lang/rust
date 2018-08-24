// ignore-tidy-linelength
// ignore-wasm32-bare unwinding being disabled causes differences in output
// ignore-wasm64-bare unwinding being disabled causes differences in output
// compile-flags: -Z verbose -Z mir-emit-validate=1

fn main() {
    let _x : Box<[i32]> = Box::new([1, 2, 3]);
}

// END RUST SOURCE
// START rustc.main.EraseRegions.after.mir
// fn main() -> () {
//     ...
//     bb1: {
//         Validate(Acquire, [_2: std::boxed::Box<[i32; 3]>]);
//         Validate(Release, [_2: std::boxed::Box<[i32; 3]>]);
//         _1 = move _2 as std::boxed::Box<[i32]> (Unsize);
//         Validate(Acquire, [_1: std::boxed::Box<[i32]>]);
//         StorageDead(_2);
//         StorageDead(_3);
//         _0 = ();
//         Validate(Release, [_1: std::boxed::Box<[i32]>]);
//         drop(_1) -> [return: bb2, unwind: bb3];
//     }
//     ...
// }
// END rustc.main.EraseRegions.after.mir
