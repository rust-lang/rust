enum Empty {}

fn empty() -> Option<Empty> {
    None
}

fn main() {
    if let Some(_x) = empty() {
        let mut _y;

        if true {
            _y = 21;
        } else {
            _y = 42;
        }

        match _x { }
    }
}

// END RUST SOURCE
// START rustc.main.UnreachablePropagation.before.mir
//      bb0: {
//          StorageLive(_1);
//          _1 = const empty() -> bb1;
//      }
//      bb1: {
//          _2 = discriminant(_1);
//          switchInt(move _2) -> [1isize: bb3, otherwise: bb2];
//      }
//      bb2: {
//          _0 = ();
//          StorageDead(_1);
//          return;
//      }
//      bb3: {
//          StorageLive(_3);
//          _3 = move ((_1 as Some).0: Empty);
//          StorageLive(_4);
//          StorageLive(_5);
//          StorageLive(_6);
//          _6 = const true;
//          switchInt(_6) -> [false: bb4, otherwise: bb5];
//      }
//      bb4: {
//          _4 = const 42i32;
//          _5 = ();
//          goto -> bb6;
//      }
//      bb5: {
//          _4 = const 21i32;
//          _5 = ();
//          goto -> bb6;
//      }
//      bb6: {
//          StorageDead(_6);
//          StorageDead(_5);
//          StorageLive(_7);
//          unreachable;
//      }
//  }
// END rustc.main.UnreachablePropagation.before.mir
// START rustc.main.UnreachablePropagation.after.mir
//      bb0: {
//          StorageLive(_1);
//          _1 = const empty() -> bb1;
//      }
//      bb1: {
//          _2 = discriminant(_1);
//          goto -> bb2;
//      }
//      bb2: {
//          _0 = ();
//          StorageDead(_1);
//          return;
//      }
//  }
// END rustc.main.UnreachablePropagation.after.mir
