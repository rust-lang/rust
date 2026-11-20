//@ needs-asm-support
//@ reference: asm.operand-type.supported-operands.sym

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
