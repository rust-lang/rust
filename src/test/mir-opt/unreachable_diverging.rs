pub enum Empty {}

fn empty() -> Option<Empty> {
    None
}

fn loop_forever() {
    loop {}
}

fn main() {
    let x = true;
    if let Some(bomb) = empty() {
        if x {
            loop_forever()
        }
        match bomb {}
    }
}

// END RUST SOURCE
// START rustc.main.UnreachablePropagation.before.mir
//      bb3: {
//          StorageLive(_4);
//          _4 = move ((_2 as Some).0: Empty);
//          StorageLive(_5);
//          StorageLive(_6);
//          _6 = _1;
//          switchInt(_6) -> [false: bb4, otherwise: bb5];
//      }
//      bb4: {
//          _5 = ();
//          goto -> bb6;
//      }
//      bb5: {
//          _5 = const loop_forever() -> bb6;
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
//      bb3: {
//          StorageLive(_4);
//          _4 = move ((_2 as Some).0: Empty);
//          StorageLive(_5);
//          StorageLive(_6);
//          _6 = _1;
//          goto -> bb4;
//      }
//      bb4: {
//          _5 = const loop_forever() -> bb5;
//      }
//      bb5: {
//          StorageDead(_6);
//          StorageDead(_5);
//          StorageLive(_7);
//          unreachable;
//      }
//  }
// END rustc.main.UnreachablePropagation.after.mir
