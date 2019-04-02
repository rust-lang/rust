// ignore-wasm32-bare compiled with panic=abort by default

fn main() {
    let mut x = Packed(Aligned(Droppy(0)));
    x.0 = Aligned(Droppy(0));
}

struct Aligned(Droppy);
#[repr(packed)]
struct Packed(Aligned);

struct Droppy(usize);
impl Drop for Droppy {
    fn drop(&mut self) {}
}

// END RUST SOURCE
// START rustc.main.EraseRegions.before.mir
// fn main() -> () {
//     let mut _0: ();
//     let mut _1: Packed;
//     let mut _2: Aligned;
//     let mut _3: Droppy;
//     let mut _4: Aligned;
//     let mut _5: Droppy;
//     let mut _6: Aligned;
//     scope 1 {
//     }
//
//     bb0: {
//         StorageLive(_1);
//         ...
//         _1 = Packed(move _2,);
//         ...
//         StorageLive(_6);
//         _6 = move (_1.0: Aligned);
//         drop(_6) -> [return: bb4, unwind: bb3];
//     }
//     bb1 (cleanup): {
//         resume;
//     }
//     bb2: {
//         StorageDead(_1);
//         return;
//     }
//     bb3 (cleanup): {
//         (_1.0: Aligned) = move _4;
//         drop(_1) -> bb1;
//     }
//     bb4: {
//         StorageDead(_6);
//         (_1.0: Aligned) = move _4;
//         StorageDead(_4);
//         _0 = ();
//         drop(_1) -> [return: bb2, unwind: bb1];
//     }
// }
// END rustc.main.EraseRegions.before.mir
