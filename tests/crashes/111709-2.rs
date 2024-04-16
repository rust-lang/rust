//@ known-bug: #111709
//@ edition: 2021

use core::arch::asm;

extern "C" fn test<T>() {}

fn uwu() {
    unsafe {
        asm!(
            "/* {0} */",
            sym test::<&mut ()>
        );
    }
}
