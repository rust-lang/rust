//@ add-minicore
//@ compile-flags: -Copt-level=0
#![feature(stmt_expr_attributes)]
#![feature(loop_match)]
#![crate_type = "lib"]

// CHECK-LABEL: runner
#[unsafe(no_mangle)]
fn runner(opcodes: &[u8]) -> i32 {
    let mut index = 0;
    let mut accum = 0;

    loop {
        // CHECK: indirectbr ptr
        #[indirect_branch]
        match unsafe { *opcodes.get_unchecked(index) } {
            b'+' => {
                accum += 1;
                index += 1;
            }
            b'-' => {
                accum -= 1;
                index += 1;
            }
            b'*' => {
                accum *= 2;
                index += 1;
            }
            0 => return accum,

            _ => return -1,
        }
    }
}
