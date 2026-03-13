//@ compile-flags: -Zmir-opt-level=0
// skip-filecheck

// Can't emit `built.after` here as that contains user type annotations which contain DefId that
// change all the time.
// EMIT_MIR uniform_array_move_out.move_out_from_end.CleanupPostBorrowck.after.mir
fn move_out_from_end() {
    let a = [Box::new(1), Box::new(2)];
    let [.., _y] = a;
}

// EMIT_MIR uniform_array_move_out.move_out_by_subslice.CleanupPostBorrowck.after.mir
fn move_out_by_subslice() {
    let a = [Box::new(1), Box::new(2)];
    let [_y @ ..] = a;
}

fn main() {
    move_out_by_subslice();
    move_out_from_end();
}
