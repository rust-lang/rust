// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Test that we correctly generate StorageDead statements for while loop
// conditions on all branches

fn get_bool(c: bool) -> bool {
    c
}

// EMIT_MIR while_storage.while_loop.PreCodegen.after.mir
fn while_loop(c: bool) {
    while get_bool(c) {
        if get_bool(c) {
            break;
        }
    }
}

fn main() {
    while_loop(false);
}
