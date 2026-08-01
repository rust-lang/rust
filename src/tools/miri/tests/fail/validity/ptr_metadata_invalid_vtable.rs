#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]
#![allow(unused_assignments)]

use core::intrinsics::mir::*;

fn main() {
    test8345(&1, |ptrptr| unsafe {
        let ptrptr = std::ptr::from_mut(ptrptr);
        ptrptr.cast::<(usize, usize)>().write((0, 1))
    });
}

trait A {}
impl<T> A for T {}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn test8345(mut ptr: &dyn A, overwrite: fn(&mut &dyn A)) {
    mir! {
        let ptrptr;
        let _unused;
        let idk;

        {
            ptrptr = &mut ptr;
            // Overwrite `ptr` to make it invalid.
            Call(_unused = overwrite(ptrptr), ReturnTo(ret), UnwindContinue())
        }

        ret = {
            // Do something with `ptr`.
            idk = PtrMetadata(ptr); //~ERROR: null reference
            Return()
        }
    }
}
