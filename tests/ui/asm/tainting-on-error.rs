//@ needs-asm-support

use std::arch::asm;

fn main() {
    unsafe {
        asm!(
            "/* {} */",
            sym None::<()>,
            //~^ ERROR invalid `sym` operand
        );
    }
}
