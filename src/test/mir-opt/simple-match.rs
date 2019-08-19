// Test that we don't generate unnecessarily large MIR for very simple matches

fn match_bool(x: bool) -> usize {
    match x {
        true => 10,
        _ => 20,
    }
}

fn main() {}


// END RUST SOURCE
// START rustc.match_bool.mir_map.0.mir
// bb0: {
//     FakeRead(ForMatchedPlace, _1);
//     switchInt(_1) -> [false: bb3, otherwise: bb2];
// }
// bb1 (cleanup): {
//     resume;
// }
// bb2: {
//     falseEdges -> [real: bb4, imaginary: bb3];
// }
// bb3: {
//     _0 = const 20usize;
//     goto -> bb5;
// }
// bb4: {
//     _0 = const 10usize;
//     goto -> bb5;
// }
// bb5: {
//     goto -> bb6;
// }
// bb6: {
//     return;
// }
// END rustc.match_bool.mir_map.0.mir
