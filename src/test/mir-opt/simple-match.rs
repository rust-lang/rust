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
//     switchInt(_1) -> [false: bb2, otherwise: bb1];
// }
// bb1: {
//     falseEdges -> [real: bb3, imaginary: bb2];
// }
// bb2: {
//     _0 = const 20usize;
//     goto -> bb4;
// }
// bb3: {
//     _0 = const 10usize;
//     goto -> bb4;
// }
// bb4: {
//     return;
// }
// END rustc.match_bool.mir_map.0.mir
