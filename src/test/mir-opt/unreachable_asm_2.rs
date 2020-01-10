// ignore-tidy-linelength
#![feature(asm)]

enum Empty {}

fn empty() -> Option<Empty> {
    None
}

fn main() {
    if let Some(_x) = empty() {
        let mut _y;

        if true {
            // asm instruction stops unreachable propagation to block bb3.
            unsafe { asm!("NOP"); }
            _y = 21;
        } else {
            // asm instruction stops unreachable propagation to block bb3.
            unsafe { asm!("NOP"); }
            _y = 42;
        }

        match _x { }
    }
}

// END RUST SOURCE
// START rustc.main.UnreachablePropagation.before.mir
//      bb3: {
//          ...
//          switchInt(_6) -> [false: bb4, otherwise: bb5];
//      }
//      bb4: {
//          StorageLive(_8);
//          asm!(InlineAsmInner { asm: "NOP", asm_str_style: Cooked, outputs: [], inputs: [], clobbers: [], volatile: true, alignstack: false, dialect: Att } : [] : []);
//          _8 = ();
//          StorageDead(_8);
//          _4 = const 42i32;
//          _5 = ();
//          goto -> bb6;
//      }
//          bb5: {
//          StorageLive(_7);
//          asm!(InlineAsmInner { asm: "NOP", asm_str_style: Cooked, outputs: [], inputs: [], clobbers: [], volatile: true, alignstack: false, dialect: Att } : [] : []);
//          _7 = ();
//          StorageDead(_7);
//          _4 = const 21i32;
//          _5 = ();
//          goto -> bb6;
//      }
//      bb6: {
//          StorageDead(_6);
//          StorageDead(_5);
//          StorageLive(_9);
//          unreachable;
//      }
//  }
// END rustc.main.UnreachablePropagation.before.mir
// START rustc.main.UnreachablePropagation.after.mir
//      bb3: {
//          ...
//          switchInt(_6) -> [false: bb4, otherwise: bb5];
//      }
//      bb4: {
//          StorageLive(_8);
//          asm!(InlineAsmInner { asm: "NOP", asm_str_style: Cooked, outputs: [], inputs: [], clobbers: [], volatile: true, alignstack: false, dialect: Att } : [] : []);
//          _8 = ();
//          StorageDead(_8);
//          _4 = const 42i32;
//          _5 = ();
//          unreachable;
//      }
//          bb5: {
//          StorageLive(_7);
//          asm!(InlineAsmInner { asm: "NOP", asm_str_style: Cooked, outputs: [], inputs: [], clobbers: [], volatile: true, alignstack: false, dialect: Att } : [] : []);
//          _7 = ();
//          StorageDead(_7);
//          _4 = const 21i32;
//          _5 = ();
//          unreachable;
//      }
//  }
// END rustc.main.UnreachablePropagation.after.mir
