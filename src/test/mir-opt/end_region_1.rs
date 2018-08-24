// compile-flags: -Z identify_regions -Z emit-end-regions
// ignore-tidy-linelength

// This is just about the simplest program that exhibits an EndRegion.

fn main() {
    let a = 3;
    let b = &a;
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
//     let mut _0: ();
//     ...
//     let _2: &'10_1rs i32;
//     ...
//     let _1: i32;
//     ...
//     bb0: {
//         StorageLive(_1);
//         _1 = const 3i32;
//         StorageLive(_2);
//         _2 = &'10_1rs _1;
//         _0 = ();
//         EndRegion('10_1rs);
//         StorageDead(_2);
//         StorageDead(_1);
//         return;
//     }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
