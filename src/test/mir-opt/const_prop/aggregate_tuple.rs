fn foo() {
    (0,).0;
    (0, 1).1;
    (0, 1, 2).2;
    (0, 1, 2, 3).3;
}

fn main() {
    foo();
    if (32u32, 16u32, 8u32).2 == 8 {
        if (1u32, 42u32).1 == 42 {
            if (2u32,).0 == 2 {
                std::process::exit(0);
            } else {
                std::process::exit(1);
            }
        } else {
            std::process::exit(2);
        }
    } else {
        std::process::exit(3);
    }
}

// END RUST SOURCE
// START rustc.foo.ConstProp.before.mir
// bb0: {
//     ...
//     _2 = (const 0i32,);
//     _1 = (_2.0: i32);
//     ...
//     _4 = (const 0i32, const 1i32);
//     _3 = (_4.1: i32);
//     ...
//     _6 = (const 0i32, const 1i32, const 2i32);
//     _5 = (_6.2: i32);
//     ...
//     _8 = (const 0i32, const 1i32, const 2i32, const 3i32);
//     _7 = (_8.3: i32);
//     ...
// }
// END rustc.foo.ConstProp.before.mir
// START rustc.foo.ConstProp.after.mir
// bb0: {
//     ...
//     _2 = const Scalar(0x00000000) : (i32,);
//     _1 = const 0i32;
//     ...
//     _4 = (const 0i32, const 1i32);
//     _3 = const 1i32;
//     ...
//     _6 = (const 0i32, const 1i32, const 2i32);
//     _5 = const 2i32;
//     ...
//     _8 = (const 0i32, const 1i32, const 2i32, const 3i32);
//     _7 = const 3i32;
//     ...
// }
// END rustc.foo.ConstProp.after.mir
// START rustc.main.ConstProp.before.mir
//  bb1: {
//      ...
//      _4 = (const 32u32, const 16u32, const 8u32);
//      _3 = (_4.2: u32);
//      _2 = Eq(move _3, const 8u32);
//      ...
//      switchInt(_2) -> [false: bb7, otherwise: bb2];
//  }
//  bb2: {
//      ...
//      _7 = (const 1u32, const 42u32);
//      _6 = (_7.1: u32);
//      _5 = Eq(move _6, const 42u32);
//      ...
//      switchInt(_5) -> [false: bb6, otherwise: bb3];
//  }
//  bb3: {
//      ...
//      _10 = (const 2u32,);
//      _9 = (_10.0: u32);
//      _8 = Eq(move _9, const 2u32);
//      ...
//      switchInt(_8) -> [false: bb5, otherwise: bb4];
//  }
//  bb4: {
//      const std::process::exit(const 0i32);
//  }
//  bb5: {
//      const std::process::exit(const 1i32);
//  }
//  bb6: {
//      const std::process::exit(const 2i32);
//  }
//  bb7: {
//      const std::process::exit(const 3i32);
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb1: {
//      ...
//      _4 = (const 32u32, const 16u32, const 8u32);
//      _3 = const 8u32;
//      _2 = const true;
//      ...
//      switchInt(const true) -> [false: bb7, otherwise: bb2];
//  }
//  bb2: {
//      ...
//      _7 = (const 1u32, const 42u32);
//      _6 = const 42u32;
//      _5 = const true;
//      ...
//      switchInt(const true) -> [false: bb6, otherwise: bb3];
//  }
//  bb3: {
//      ...
//      _10 = const Scalar(0x00000002) : (u32,);
//      _9 = const 2u32;
//      _8 = const true;
//      ...
//      switchInt(const true) -> [false: bb5, otherwise: bb4];
//  }
//  bb4: {
//      const std::process::exit(const 0i32);
//  }
//  bb5: {
//      const std::process::exit(const 1i32);
//  }
//  bb6: {
//      const std::process::exit(const 2i32);
//  }
//  bb7: {
//      const std::process::exit(const 3i32);
//  }
// END rustc.main.ConstProp.after.mir
// START rustc.main.SimplifyCfg-final.after.mir
// bb1: {
//     ...
//     (_4.0: u32) = const 32u32;
//     (_4.1: u32) = const 16u32;
//     (_4.2: u32) = const 8u32;
//     _3 = const 8u32;
//     _2 = const true;
//     ...
//     (_7.0: u32) = const 1u32;
//     (_7.1: u32) = const 42u32;
//     _6 = const 42u32;
//     _5 = const true;
//     ...
//     _10 = const Scalar(0x00000002) : (u32,);
//     _9 = const 2u32;
//     _8 = const true;
//     ...
//     const std::process::exit(const 0i32);
// }
// END rustc.main.SimplifyCfg-final.after.mir
