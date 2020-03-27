// ignore-tidy-linelength
#![feature(llvm_asm)]

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

        // asm instruction stops unreachable propagation to if else blocks bb4 and bb5.
        unsafe { llvm_asm!("NOP"); }
        match _x { }
    }
}

// END RUST SOURCE
// START rustc.main.UnreachablePropagation.before.mir
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
//          llvm_asm!(LlvmInlineAsmInner { asm: "NOP", asm_str_style: Cooked, outputs: [], inputs: [], clobbers: [], volatile: true, alignstack: false, dialect: Att } : [] : []);
//          _7 = ();
//          StorageDead(_7);
//          StorageLive(_8);
//          unreachable;
//      }
//  }
// END rustc.main.UnreachablePropagation.before.mir
// START rustc.main.UnreachablePropagation.after.mir
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
//          llvm_asm!(LlvmInlineAsmInner { asm: "NOP", asm_str_style: Cooked, outputs: [], inputs: [], clobbers: [], volatile: true, alignstack: false, dialect: Att } : [] : []);
//          _7 = ();
//          StorageDead(_7);
//          StorageLive(_8);
//          unreachable;
//      }
//  }
// END rustc.main.UnreachablePropagation.after.mir
