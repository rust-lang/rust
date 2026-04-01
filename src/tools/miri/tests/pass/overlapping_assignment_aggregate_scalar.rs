#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime")]
fn main() {
    mir! {
        let _1: (u8,);
        {
            _1.0 = 0_u8;
            // This is a scalar type, so overlap is (for now) not UB.
            // However, we used to treat such overlapping assignments incorrectly
            // (see <https://github.com/rust-lang/rust/issues/146383#issuecomment-3273224645>).
            _1 = (_1.0, );
            Return()
        }
    }
}
