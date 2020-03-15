#![feature(generators, generator_trait)]

// Regression test for #58892, generator drop shims should not have blocks
// spuriously marked as cleanup

fn main() {
    let gen = || {
        let _s = String::new();
        yield;
    };
}

// END RUST SOURCE

// START rustc.main-{{closure}}.generator_drop.0.mir
// bb0: {
//     _9 = discriminant((*_1));
//     switchInt(move _9) -> [0u32: bb7, 3u32: bb11, otherwise: bb12];
// }
// bb1 (cleanup): {
//     resume;
// }
// bb2 (cleanup): {
//     nop;
//     goto -> bb8;
// }
// bb3: {
//     StorageDead(_5);
//     StorageDead(_4);
//     drop((((*_1) as variant#3).0: std::string::String)) -> [return: bb4, unwind: bb2];
// }
// bb4: {
//     nop;
//     goto -> bb9;
// }
// bb5: {
//     return;
// }
// bb6: {
//     return;
// }
// bb7: {
//     goto -> bb10;
// }
// bb8 (cleanup): {
//     goto -> bb1;
// }
// bb9: {
//     goto -> bb5;
// }
// bb10: {
//     goto -> bb6;
// }
// bb11: {
//     StorageLive(_4);
//     StorageLive(_5);
//     goto -> bb3;
// }
// bb12: {
//     return;
// }
// END rustc.main-{{closure}}.generator_drop.0.mir
