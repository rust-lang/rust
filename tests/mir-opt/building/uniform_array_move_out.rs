//@ compile-flags: -Zmir-opt-level=0
//@ needs-unwind
// skip-filecheck
#![feature(liballoc_internals, rustc_attrs)]

// EMIT_MIR uniform_array_move_out.move_out_from_end.ElaborateDrops.diff
fn move_out_from_end() {
    let a = [Box::new(1), Box::new(2)];
    let [.., _y] = a;
}

// EMIT_MIR uniform_array_move_out.move_out_from_middle.ElaborateDrops.diff
fn move_out_from_middle() {
    let a = [Box::new(1), Box::new(2), Box::new(3)];
    let [_, _y, _] = a;
}

// EMIT_MIR uniform_array_move_out.move_out_by_subslice.ElaborateDrops.diff
fn move_out_by_subslice() {
    let a = [Box::new(1), Box::new(2), Box::new(3)];
    let [_y @ .., _] = a;
}

fn main() {
    move_out_by_subslice();
    move_out_from_end();
}
