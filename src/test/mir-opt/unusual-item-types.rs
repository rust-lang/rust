// Test that we don't ICE when trying to dump MIR for unusual item types and
// that we don't create filenames containing `<` and `>`
// ignore-tidy-linelength

struct A;

impl A {
    const ASSOCIATED_CONSTANT: i32 = 2;
}

// See #59021
enum Test {
    X(usize),
    Y { a: usize },
}

enum E {
    V = 5,
}

fn main() {
    let f = Test::X as fn(usize) -> Test;
    let v = Vec::<i32>::new();
}

// END RUST SOURCE

// START rustc.{{impl}}-ASSOCIATED_CONSTANT.mir_map.0.mir
// bb0: {
//     _0 = const 2i32;
//     return;
// }
// bb1 (cleanup): {
//     resume;
// }
// END rustc.{{impl}}-ASSOCIATED_CONSTANT.mir_map.0.mir

// START rustc.E-V-{{constant}}.mir_map.0.mir
// bb0: {
//     _0 = const 5isize;
//     return;
// }
// bb1 (cleanup): {
//     resume;
// }
// END rustc.E-V-{{constant}}.mir_map.0.mir

// START rustc.ptr-real_drop_in_place.std__vec__Vec_i32_.AddMovesForPackedDrops.before.mir
//     bb0: {
//     goto -> bb7;
// }
// bb1: {
//     return;
// }
// bb2 (cleanup): {
//     resume;
// }
// bb3: {
//     goto -> bb1;
// }
// bb4 (cleanup): {
//     goto -> bb2;
// }
// bb5 (cleanup): {
//     drop(((*_1).0: alloc::raw_vec::RawVec<i32>)) -> bb4;
// }
// bb6: {
//     drop(((*_1).0: alloc::raw_vec::RawVec<i32>)) -> [return: bb3, unwind: bb4];
// }
// bb7: {
//     _2 = &mut (*_1);
//     _3 = const <std::vec::Vec<i32> as std::ops::Drop>::drop(move _2) -> [return: bb6, unwind: bb5];
// }
// END rustc.ptr-real_drop_in_place.std__vec__Vec_i32_.AddMovesForPackedDrops.before.mir

// START rustc.Test-X-{{constructor}}.mir_map.0.mir
// fn Test::X(_1: usize) -> Test {
//     let mut _0: Test;
//
//     bb0: {
//         ((_0 as X).0: usize) = move _1;
//         discriminant(_0) = 0;
//         return;
//     }
// }
// END rustc.Test-X-{{constructor}}.mir_map.0.mir
